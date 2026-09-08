"""
S3R / TSR-Open 部署推理引擎（不修改训练脚本）。

从训练输出目录加载:
  - evaluation_results.json
  - best.weights.h5
  - s3r_centers_*.npy / s3r_thetas_*.npy / s3r_inv_covs_*.npy
  - deploy_manifest.json（可选，由 export_s3r_deploy_bundle.py 生成）

推断规则与 work_V2_S3R_semantics_loss_distance_compare.py 一致:
  Rim-3σ per-class Mahalanobis + margin_bias + 可选重建门控。
"""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_ROOT)

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

import work_2_CMSDTN_TCN_model as cms
import work_2_TFD_IBG_model as tfd
import work_V2_S3R_semantics_loss_distance_compare as w


@dataclass
class S3RPrediction:
    """单样本开放集推断结果。"""

    is_known: bool
    class_id: int
    class_name: str
    min_margin: float
    distances: np.ndarray
    margins_per_class: np.ndarray
    model_probs: np.ndarray | None = None
    recon_error: float | None = None
    recon_pass: bool | None = None


@dataclass
class S3RDetectorConfig:
    run_dir: str
    eval_cfg: dict
    manifest: dict | None = None
    metric: str = 'mahalanobis'
    margin_bias: float = 0.0
    recon_stats: dict | None = None
    recon_k: float = 3.0
    known_classes: list[str] = field(default_factory=list)
    input_shape: tuple[int, ...] = (320, 512, 2)


def _read_json(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _first_existing(run_dir: str, names: tuple[str, ...]) -> str | None:
    for name in names:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    return None


def resolve_run_dir(run_dir: str) -> str:
    run_dir = os.path.abspath(run_dir)
    eval_path = os.path.join(run_dir, 'evaluation_results.json')
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(f'缺少 evaluation_results.json: {run_dir}')
    weights = _first_existing(run_dir, ('best.weights.h5', 'best.keras', 'model.keras'))
    if weights is None:
        raise FileNotFoundError(f'缺少权重文件 (best.weights.h5): {run_dir}')
    return run_dir


def _infer_tfd_kwargs(eval_cfg: dict, manifest: dict | None) -> dict[str, Any]:
    """从 deploy_manifest 或 run_subdir 推断 TFD 主干开关（与训练时变体对齐）。"""
    if manifest and 'tfd' in manifest:
        return dict(manifest['tfd'])

    sub = str(eval_cfg.get('run_subdir') or '')
    parts = {p for p in sub.replace('\\', '/').split('/') if p}

    kwargs = {
        'freq_k': w.TFD_FREQ_K,
        'time_k': w.TFD_TIME_K,
        'ibg_per_channel': w.TFD_IBG_PER_CHANNEL,
        'use_joint': w.TFD_USE_JOINT,
        'use_ibg': w.TFD_USE_IBG,
        'ibg_mode': w.TFD_IBG_MODE,
        'ibg_gate_scale': w.TFD_IBG_GATE_SCALE,
        'use_fpe': w.TFD_USE_FPE,
        'fpe_mode': w.TFD_FPE_MODE,
    }

    if 'tfd_fpe' in parts:
        kwargs.update(use_fpe=True, use_ibg=False)
    elif 'tfd_nofpe' in parts:
        kwargs.update(use_fpe=False, use_ibg=False)
    elif 'tfd_noibg' in parts:
        kwargs.update(use_ibg=False)
    elif 'ibg_on' in parts:
        kwargs.update(use_ibg=True)
    elif 'ibg_off' in parts:
        kwargs.update(use_ibg=False)
    elif 'tfd_resid_ibg' in parts:
        kwargs.update(use_ibg=True, ibg_mode='residual')

    return kwargs


def _loss_cfg_from_eval(eval_cfg: dict) -> dict | None:
    loss_name = eval_cfg['loss_name']
    if not w.is_tsr_open_loss(loss_name):
        return None
    preset = w.tsr_preset_for_loss_name(loss_name)
    cfg = w.get_tsr_loss_cfg(preset)
    train_rec = eval_cfg.get('s3r_train') or {}
    if 'enable_reconstruction' in train_rec:
        cfg = dict(cfg)
        if not train_rec['enable_reconstruction']:
            cfg['lambda_rec'] = 0.0
    return cfg


def _enable_reconstruction(eval_cfg: dict, loss_cfg: dict | None) -> bool:
    train_rec = eval_cfg.get('s3r_train') or {}
    if 'enable_reconstruction' in train_rec:
        return bool(train_rec['enable_reconstruction'])
    loss_name = eval_cfg['loss_name']
    return w.loss_needs_reconstruction(loss_name, loss_cfg)


def build_semantic_model(eval_cfg: dict, manifest: dict | None = None, *, sample_batch=None):
    """按 evaluation_results 重建 SemanticOpenSetModel（与 run_one_loss_experiment 一致）。"""
    loss_name = eval_cfg['loss_name']
    if loss_name not in w.LOSS_BUILDERS:
        raise ValueError(f'未知 loss_name={loss_name!r}，请确认训练脚本已注册该损失')

    loss_cfg = _loss_cfg_from_eval(eval_cfg)
    l2_norm = bool(eval_cfg.get('l2_normalize_embedding', False))
    embedding_dim = int(eval_cfg.get('embedding_dim', w.EMBEDDING_DIM))
    input_shape = tuple(eval_cfg.get('input_shape', (320, 512, 2)))
    backbone = eval_cfg.get('backbone_preset', w.BACKBONE_PRESET)
    enable_recon = _enable_reconstruction(eval_cfg, loss_cfg)

    if backbone == 'tfd_ibg':
        tfd_kw = _infer_tfd_kwargs(eval_cfg, manifest)
        base = tfd.build_tfd_ibg_open_set_model(
            input_shape=input_shape,
            num_known_classes=w.NUM_KNOWN_CLASSES,
            embedding_dim=embedding_dim,
            l2_normalize_embedding=l2_norm,
            distance_temperature=w.DISTANCE_TEMPERATURE,
            enable_reconstruction=enable_recon,
            freq_k=tfd_kw['freq_k'],
            time_k=tfd_kw['time_k'],
            ibg_per_channel=tfd_kw['ibg_per_channel'],
            ibg_mode=tfd_kw['ibg_mode'],
            ibg_gate_scale=tfd_kw['ibg_gate_scale'],
            use_joint=tfd_kw['use_joint'],
            use_ibg=tfd_kw['use_ibg'],
            use_fpe=tfd_kw['use_fpe'],
            fpe_mode=tfd_kw['fpe_mode'],
        )
    else:
        base = cms.open_set_single_head_model(
            input_shape=input_shape,
            num_known_classes=w.NUM_KNOWN_CLASSES,
            embedding_dim=embedding_dim,
            backbone_preset=backbone,
            hybrid_last_dilation=int(eval_cfg.get('hybrid_last_dilation', w.HYBRID_LAST_DILATION)),
            hybrid_ms_branch3_w_dilation=int(
                eval_cfg.get('hybrid_ms_branch3_w_dilation', w.HYBRID_MS_BRANCH3_W_DILATION)
            ),
            l2_normalize_embedding=l2_norm,
            distance_temperature=w.DISTANCE_TEMPERATURE,
            enable_reconstruction=enable_recon,
            prototype_conditioned_recon=False,
            enable_stora=False,
        )

    model = w.SemanticOpenSetModel(
        model=base,
        num_known_classes=w.NUM_KNOWN_CLASSES,
        auto_threshold=True,
        mahalanobis_ridge=w.MAHALANOBIS_RIDGE,
    )
    model.use_embedding_loss = w.is_tsr_open_loss(loss_name) or loss_name == 's3r_paper'
    model.use_anchor_aux = (
        (w.is_tsr_open_loss(loss_name) and loss_cfg is not None and loss_cfg.get('lambda_metric', 0) > 0)
        or (
            loss_name == 's3r_paper'
            and (w.S3R_LAMBDA_ANC > 0 or w.S3R_MP_INTER_LAMBDA > 0)
        )
    )
    model.compile(
        optimizer=Adam(w.INITIAL_LR),
        model_loss=w.LOSS_BUILDERS[loss_name](w.NUM_KNOWN_CLASSES),
    )
    if sample_batch is not None:
        _ = model(sample_batch, training=False)
    return model


def load_s3r_state_from_run_dir(run_dir: str, metric: str = 'mahalanobis') -> dict:
    """从训练输出目录加载已拟合的 Rim-3σ 边界（优先于重新拟合）。"""
    centers_path = os.path.join(run_dir, f's3r_centers_{metric}.npy')
    thetas_path = os.path.join(run_dir, f's3r_thetas_{metric}.npy')
    if not os.path.isfile(centers_path) or not os.path.isfile(thetas_path):
        raise FileNotFoundError(
            f'缺少 S3R 边界文件: {centers_path} 或 {thetas_path}\n'
            '请等训练完成开放集评估后再部署，或指定含训练数据的 data_dir 以自动拟合'
        )
    centers = np.load(centers_path).astype(np.float64)
    thetas = np.load(thetas_path).astype(np.float64)
    inv_covs = None
    inv_path = os.path.join(run_dir, f's3r_inv_covs_{metric}.npy')
    if metric == 'mahalanobis':
        if not os.path.isfile(inv_path):
            raise FileNotFoundError(f'缺少马氏协方差逆矩阵: {inv_path}')
        inv_covs = np.load(inv_path).astype(np.float64)

    return {
        'metric': metric,
        'centers': centers,
        'inv_covs': inv_covs,
        'thetas': thetas,
        'thetas_raw': thetas.copy(),
        'theta_global_scale': 1.0,
        'theta_mp_scale': 1.0,
        'ridge': float(w.MAHALANOBIS_RIDGE),
        'sigma_factor': float(w.S3R_SIGMA_FACTOR),
        'center_source': 'saved_artifacts',
    }


def _margin_bias_from_eval(eval_cfg: dict, metric: str) -> float:
    inf = eval_cfg.get('s3r_inference') or {}
    biases = inf.get('margin_biases') or {}
    if metric in biases:
        return float(biases[metric])
    dc = (eval_cfg.get('distance_comparison') or {}).get(metric) or {}
    if 'margin_bias' in dc:
        return float(dc['margin_bias'])
    return 0.0


def _recon_stats_from_run_dir(run_dir: str, eval_cfg: dict) -> dict | None:
    path = os.path.join(run_dir, 'reconstruction_stats.json')
    if os.path.isfile(path):
        return _read_json(path)
    recon_block = eval_cfg.get('inference_recon_gate') or {}
    mah = (eval_cfg.get('distance_comparison') or {}).get('mahalanobis') or {}
    gate = mah.get('open_set_detection', {}).get('recon_gate')
    if gate:
        return {'mean': gate['mean'], 'std': gate['std'], 'k': gate.get('k', w.S3R_RECON_K)}
    return None


def prepare_batch(x: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    """
    将输入整理为 (N, H, W, C)，与 OpenSetDataGenerator 一致。
    支持: (H,W,C), (C,H,W), (N,C,H,W), (N,H,W,C)
    """
    arr = np.asarray(x, dtype=np.float32)
    h, w, c = input_shape
    if arr.ndim == 3:
        if arr.shape == (c, h, w):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape != (h, w, c):
            raise ValueError(f'单样本形状 {arr.shape} 与模型 input_shape {input_shape} 不匹配')
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 4:
        if arr.shape[1:] == (c, h, w):
            arr = np.transpose(arr, (0, 2, 3, 1))
        elif arr.shape[1:] != (h, w, c):
            raise ValueError(f'批次形状 {arr.shape} 与模型 input_shape {input_shape} 不匹配')
    else:
        raise ValueError(f'期望 3D 或 4D 输入，收到 shape={arr.shape}')
    return arr


class S3RDetector:
    """可嵌入任意软件的 S3R 开放集检测器。"""

    def __init__(self, run_dir: str, *, metric: str | None = None, lazy: bool = False):
        self.run_dir = resolve_run_dir(run_dir)
        self.eval_path = os.path.join(self.run_dir, 'evaluation_results.json')
        self.eval_cfg = _read_json(self.eval_path)
        manifest_path = os.path.join(self.run_dir, 'deploy_manifest.json')
        self.manifest = _read_json(manifest_path) if os.path.isfile(manifest_path) else None

        inf = self.eval_cfg.get('s3r_inference') or {}
        self.metric = metric or inf.get('primary_metric', 'mahalanobis')
        self.known_classes = list(self.eval_cfg.get('known_classes') or [])
        self.input_shape = tuple(self.eval_cfg.get('input_shape', (320, 512, 2)))
        self.margin_bias = _margin_bias_from_eval(self.eval_cfg, self.metric)
        self.recon_stats = _recon_stats_from_run_dir(self.run_dir, self.eval_cfg)
        self.recon_k = float((self.recon_stats or {}).get('k', w.S3R_RECON_K))

        self.model = None
        self.state = None
        if not lazy:
            self.load()

    @classmethod
    def from_run_dir(cls, run_dir: str, **kwargs) -> 'S3RDetector':
        return cls(run_dir, **kwargs)

    def load(self) -> None:
        """加载权重与 Rim-3σ 边界。"""
        cms.K.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(int(self.eval_cfg.get('random_seed', 42)))

        dummy = np.zeros((1, *self.input_shape), dtype=np.float32)
        self.model = build_semantic_model(self.eval_cfg, self.manifest, sample_batch=dummy)

        weights = _first_existing(self.run_dir, ('best.weights.h5',))
        if weights is None:
            raise FileNotFoundError(f'未找到 best.weights.h5: {self.run_dir}')
        self.model.load_weights(weights)
        self.state = load_s3r_state_from_run_dir(self.run_dir, metric=self.metric)

    def predict_batch(self, x: np.ndarray) -> list[S3RPrediction]:
        if self.model is None or self.state is None:
            self.load()

        batch = prepare_batch(x, self.input_shape)
        out = self.model.model(batch, training=False)
        emb = out['embedding'].numpy().astype(np.float64)
        is_known, pred_cls, min_margin = w.s3r_decision_from_embedding(
            emb, self.state, margin_bias=self.margin_bias,
        )

        use_recon = (
            self.recon_stats is not None
            and float(self.recon_stats.get('std', 0)) > 1e-9
            and 'reconstructed' in out
        )
        rec_mean = float(self.recon_stats['mean']) if use_recon else 0.0
        rec_std = float(self.recon_stats['std']) if use_recon else 1.0
        rec_thresh = rec_mean + self.recon_k * rec_std
        recon_errors = None
        recon_ok = None
        if use_recon:
            diff = batch - out['reconstructed'].numpy()
            recon_errors = np.sqrt(np.sum(diff ** 2, axis=(1, 2, 3)) + 1e-8)
            recon_ok = recon_errors <= rec_thresh
            is_known = is_known & recon_ok

        model_probs = None
        if 'model_output' in out:
            model_probs = out['model_output'].numpy()

        d_mat = w.s3r_distances_to_all_classes(emb, self.state)
        margins = d_mat - self.state['thetas'].reshape(1, -1)

        results: list[S3RPrediction] = []
        for i in range(batch.shape[0]):
            cid = int(pred_cls[i])
            cname = self.known_classes[cid] if 0 <= cid < len(self.known_classes) else 'UNKNOWN'
            results.append(
                S3RPrediction(
                    is_known=bool(is_known[i]),
                    class_id=cid,
                    class_name=cname if is_known[i] else 'UNKNOWN',
                    min_margin=float(min_margin[i]),
                    distances=d_mat[i].copy(),
                    margins_per_class=margins[i].copy(),
                    model_probs=model_probs[i].copy() if model_probs is not None else None,
                    recon_error=float(recon_errors[i]) if recon_errors is not None else None,
                    recon_pass=bool(recon_ok[i]) if recon_ok is not None else None,
                )
            )
        return results

    def predict(self, x: np.ndarray) -> S3RPrediction:
        return self.predict_batch(x)[0]

    def summary(self) -> S3RDetectorConfig:
        return S3RDetectorConfig(
            run_dir=self.run_dir,
            eval_cfg=self.eval_cfg,
            manifest=self.manifest,
            metric=self.metric,
            margin_bias=self.margin_bias,
            recon_stats=self.recon_stats,
            recon_k=self.recon_k,
            known_classes=self.known_classes,
            input_shape=self.input_shape,
        )


def format_prediction(pred: S3RPrediction) -> str:
    lines = [
        f"判定: {'已知无人机' if pred.is_known else '未知信号'}",
        f"型号: {pred.class_name}",
        f"min_margin: {pred.min_margin:.4f}",
    ]
    if pred.model_probs is not None:
        top = int(np.argmax(pred.model_probs))
        lines.append(f"CE 头 argmax: {top} (p={pred.model_probs[top]:.4f})")
    if pred.recon_error is not None:
        lines.append(f"重建误差: {pred.recon_error:.4f} ({'通过' if pred.recon_pass else '拒绝'})")
    return '\n'.join(lines)

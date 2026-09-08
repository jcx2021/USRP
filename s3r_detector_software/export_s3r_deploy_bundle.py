"""
将训练完成的 run 目录打包为可部署 bundle（不修改训练脚本）。

用法（工作目录 DroneDetect_V2）:
  python s3r_detector_software/export_s3r_deploy_bundle.py model_s3r_open_set_result/.../seed42
  python -m s3r_detector_software.export_s3r_deploy_bundle <run_dir> --out deploy/my_detector
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

if __package__ in (None, ''):
    _pkg = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_pkg)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)

import work_V2_S3R_semantics_loss_distance_compare as w
from s3r_inference_engine import _infer_tfd_kwargs, _read_json, resolve_run_dir

ARTIFACT_NAMES = (
    'evaluation_results.json',
    'best.weights.h5',
    'model_summary.txt',
    'model_flops.txt',
    's3r_centers_mahalanobis.npy',
    's3r_thetas_mahalanobis.npy',
    's3r_inv_covs_mahalanobis.npy',
    's3r_centers_euclidean.npy',
    's3r_thetas_euclidean.npy',
    's3r_margin_bias_mahalanobis.npy',
    's3r_margin_bias_euclidean.npy',
)


def build_manifest(eval_cfg: dict) -> dict:
    manifest = {
        'source': 'work_V2_S3R_semantics_loss_distance_compare.py',
        'loss_name': eval_cfg.get('loss_name'),
        'backbone_preset': eval_cfg.get('backbone_preset'),
        'run_subdir': eval_cfg.get('run_subdir'),
        'random_seed': eval_cfg.get('random_seed'),
        'known_classes': eval_cfg.get('known_classes'),
        'input_shape': eval_cfg.get('input_shape'),
        's3r_inference': eval_cfg.get('s3r_inference'),
    }
    if eval_cfg.get('backbone_preset') == 'tfd_ibg':
        manifest['tfd'] = _infer_tfd_kwargs(eval_cfg, None)
    train = eval_cfg.get('s3r_train') or {}
    if train.get('enable_reconstruction'):
        manifest['enable_reconstruction'] = True
    return manifest


def export_bundle(run_dir: str, out_dir: str) -> str:
    run_dir = resolve_run_dir(run_dir)
    eval_cfg = _read_json(os.path.join(run_dir, 'evaluation_results.json'))
    os.makedirs(out_dir, exist_ok=True)

    copied = []
    for name in ARTIFACT_NAMES:
        src = os.path.join(run_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))
            copied.append(name)

    manifest = build_manifest(eval_cfg)
    manifest['copied_files'] = copied
    manifest_path = os.path.join(out_dir, 'deploy_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    recon = None
    for key in ('inference_recon_gate',):
        block = eval_cfg.get(key)
        if block and 'mahalanobis' in block:
            recon = block['mahalanobis']
            break
    if recon is None:
        mah = (eval_cfg.get('distance_comparison') or {}).get('mahalanobis') or {}
        gate = (mah.get('open_set_detection') or {}).get('recon_gate')
        if gate:
            recon = gate
    if recon:
        with open(os.path.join(out_dir, 'reconstruction_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(recon, f, indent=2)

    print(f'已导出到: {out_dir}')
    print(f'  复制文件: {len(copied)} 个')
    print(f'  损失: {eval_cfg.get("loss_name")}  骨干: {eval_cfg.get("backbone_preset")}')
    return out_dir


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    parser = argparse.ArgumentParser(description='导出 S3R 部署 bundle')
    parser.add_argument('run_dir', help='训练输出目录（含 evaluation_results.json）')
    parser.add_argument(
        '--out',
        default=None,
        help='输出目录（默认: <run_dir>_deploy）',
    )
    args = parser.parse_args()
    out = args.out or f'{os.path.abspath(args.run_dir).rstrip(os.sep)}_deploy'
    export_bundle(args.run_dir, out)


if __name__ == '__main__':
    main()

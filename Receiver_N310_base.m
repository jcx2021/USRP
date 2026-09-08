%% ============================================================
%  USRP N310 高速采集 + 复信号STFT可视化
%% ============================================================

clear; clc;

params.centerFreq      = 2.4375e9;
params.gain            = 50;              % 固定接收增益 (dB)
params.sampleRate      = 60e6;            % 目标采样率 60 MHz（精确）
params.masterClockRate = 122.88e6;        % N310 最接近 60M 的硬件主时钟
params.decimFactor     = 2;
params.hwSampleRate    = params.masterClockRate / params.decimFactor; % 61.44 MHz
% 精确有理数换算: 60/61.44 = 125/128，N310 上能达到的最准 60 MHz 方案
params.resampleP       = 125;
params.resampleQ       = 128;
params.resampleFilter  = 160;             % 抗混叠 FIR 阶数，越大频响越准、越慢
params.numSamples      = 120000000;       % 60 MHz 下约 2 s（须为 125 的整数倍）
params.hwNumSamples    = params.numSamples * params.resampleQ / params.resampleP;
params.dataRoot        = 'E:\USRP';       % 数据根目录（改盘符只改这一行）
params.formalOutputDir = fullfile(params.dataRoot, 'recordings');
params.testOutputDir   = fullfile(params.dataRoot, 'test');
params.recordDuration  = params.numSamples / params.sampleRate;
params.segDuration     = 0.1;            % 时频图每片段时长 (s)
params.numSegPlots     = 3;               % 时频图输出片段数（在 2s 内均匀取）
params.stftWindow      = 1024;            % STFT 窗长
params.stftHop         = 512;             % STFT 帧移（间隔）
params.stftNfft        = 1024;            % STFT FFT 点数

assert(mod(params.numSamples, params.resampleP) == 0, ...
    'numSamples 须为 %d 的整数倍以保证精确对齐', params.resampleP);

%% 采集任务配置（采集前先输入）
fprintf('=== 采集任务配置 ===\n');
isTestAnswer = lower(strtrim(input('是否为测试采集？(y/n): ', 's')));
isTestMode = ismember(isTestAnswer, {'y', 'yes', '是', '1'});
stateLabels = {'ON', 'HO', 'FY'};
stateDesc   = {'开机不起飞', '起飞悬停', '平移'};

if isTestMode
    params.outputDir = params.testOutputDir;
    if ~exist(params.outputDir, 'dir'), mkdir(params.outputDir); end
    stateDir = params.outputDir;
    droneModel  = 'TEST';
    droneID     = '';
    flightState = 'TEST';
    stateIdx    = 0;
    fprintf('模式: 测试 → 保存至 %s（无需命名，自动编号）\n', params.outputDir);
else
    params.outputDir = params.formalOutputDir;
    fprintf('模式: 正式 → 保存至 %s\n', params.outputDir);
    droneModel = upper(strtrim(input('无人机型号 (如 PHA): ', 's')));
    droneID    = strtrim(input('个体 ID (如 01): ', 's'));
    fprintf('飞行状态:\n');
    for k = 1:numel(stateLabels)
        fprintf('  %d = %s (%s)\n', k, stateLabels{k}, stateDesc{k});
    end
    stateIdx = input('请选择状态编号 [1-3]: ');
    if isempty(stateIdx) || stateIdx < 1 || stateIdx > 3 || stateIdx ~= floor(stateIdx)
        error('无效状态编号，请输入 1、2 或 3。');
    end
    flightState = stateLabels{stateIdx};
    if ~exist(params.outputDir, 'dir'), mkdir(params.outputDir); end
    individualDir = fullfile(params.outputDir, sprintf('%s_%s', droneModel, droneID));
    if ~exist(individualDir, 'dir'), mkdir(individualDir); end
    for k = 1:numel(stateLabels)
        stateSubDir = fullfile(individualDir, stateLabels{k});
        if ~exist(stateSubDir, 'dir'), mkdir(stateSubDir); end
    end
    stateDir = fullfile(individualDir, flightState);
    fprintf('个体目录: %s\n', individualDir);
    fprintf('本次状态: %s (%s)\n', flightState, stateDesc{stateIdx});
end

existing = dir(fullfile(stateDir, '*.dat'));
recordIndex = 0;
for k = 1:numel(existing)
    idxVal = str2double(existing(k).name(1:end-4));
    if ~isnan(idxVal) && idxVal == floor(idxVal) && idxVal >= 0
        recordIndex = max(recordIndex, idxVal + 1);
    end
end
savePath = fullfile(stateDir, sprintf('%d.dat', recordIndex));
fprintf('将保存为: %s\n', savePath);

fprintf('\n=== USRP N310 高速采集 ===\n');
fprintf('增益: %.0f dB | 中心频率: %.4f GHz\n', params.gain, params.centerFreq/1e9);
fprintf('硬件: %.2f MHz → 重采样 %d/%d → %.0f MHz（精确有理比）\n', ...
    params.hwSampleRate/1e6, params.resampleP, params.resampleQ, params.sampleRate/1e6);

%% 释放残留 USRP 连接（上次异常退出时可能未 release）
releaseStaleSDRuReceivers();

%% 创建接收对象
rx = [];
try
    rx = comm.SDRuReceiver('Platform', 'N310', ...
        'IPAddress', '192.168.20.2', ...
        'CenterFrequency', params.centerFreq, ...
        'Gain', params.gain, ...
        'MasterClockRate', params.masterClockRate, ...
        'DecimationFactor', params.decimFactor, ...
        'SamplesPerFrame', 375000, ...
        'TransportDataType', 'int16', ...
        'OutputDataType', 'double');

    %% 预热
    for i = 1:50, [~,~,~] = step(rx); end

    %% 采集（硬件 61.44 MHz）
    allData_hw = complex(zeros(params.hwNumSamples, 1));
    idx = 1;
    tic;
    while idx <= params.hwNumSamples
        [frame, len, ~] = step(rx);
        if len > 0
            n = min(len, params.hwNumSamples - idx + 1);
            allData_hw(idx:idx+n-1) = frame(1:n);
            idx = idx + n;
        end
    end
    fprintf('硬件采集完成: %.2f s\n', toc);
catch ME
    releaseStaleSDRuReceivers();
    rethrow(ME);
end
releaseStaleSDRuReceivers();

%% 重采样至 60 MHz（61.44 × 125/128 = 60，精确）
allData = resample(allData_hw, params.resampleP, params.resampleQ, params.resampleFilter);
allData = allData(1:params.numSamples);
fprintf('重采样完成: %d → %d 样本 @ %.0f MHz\n', ...
    length(allData_hw), length(allData), params.sampleRate/1e6);
printSignalQualityReport(allData);

%% 保存 .dat（I/Q 交错 float32 小端）
I_data = single(real(allData));
Q_data = single(imag(allData));
interleaved = zeros(2*length(allData), 1, 'single');
interleaved(1:2:end) = I_data;
interleaved(2:2:end) = Q_data;
fid = fopen(savePath, 'wb');
fwrite(fid, interleaved, 'single', 'l');
fclose(fid);
fileInfo = dir(savePath);
fprintf('已保存: %s (%.1f MB, %d 样本, %.1f s @ %.0f MHz)\n', ...
    savePath, fileInfo.bytes/(1024*1024), length(allData), ...
    params.recordDuration, params.sampleRate/1e6);

%% ==================== 可视化 ====================

% 去直流
allData_clean = allData - mean(allData);
I_clean = real(allData_clean);
Q_clean = imag(allData_clean);

%% ========== 图1：单帧数据 I/Q时域 + 复信号FFT ==========
fig1 = figure('Name', '单帧数据 - 复信号分析', 'Position', [50 50 1600 800], 'Color', 'w');

frameSamples = round(375000 * params.sampleRate / params.hwSampleRate);
t_frame = (0:frameSamples-1) / params.sampleRate * 1000;  % ms

% I路时域
subplot(2, 3, 1);
plot(t_frame, I_clean(1:frameSamples), 'Color', [0.2 0.4 0.8], 'LineWidth', 0.5);
xlabel('时间 / ms'); ylabel('幅度');
title('I路 - 时域波形'); grid on; xlim([0 t_frame(end)]);

% Q路时域
subplot(2, 3, 2);
plot(t_frame, Q_clean(1:frameSamples), 'Color', [0.8 0.3 0.2], 'LineWidth', 0.5);
xlabel('时间 / ms'); ylabel('幅度');
title('Q路 - 时域波形'); grid on; xlim([0 t_frame(end)]);

% 复信号幅度时域
subplot(2, 3, 3);
amp_frame = abs(allData_clean(1:frameSamples));
plot(t_frame, amp_frame, 'Color', [0.3 0.6 0.3], 'LineWidth', 0.5);
xlabel('时间 / ms'); ylabel('幅度');
title('复信号幅度 |I+jQ|'); grid on; xlim([0 t_frame(end)]);

% I路FFT（实信号，对称）
subplot(2, 3, 4);
Nfft = 2^19;
Y_I = fftshift(fft(I_clean(1:min(Nfft, length(I_clean)))));
f = (-length(Y_I)/2:length(Y_I)/2-1) * (params.sampleRate / length(Y_I));
plot(f/1e6, 10*log10(abs(Y_I).^2 + eps), 'Color', [0.2 0.4 0.8], 'LineWidth', 0.8);
xlim([-30 30]); ylim([-80 40]);
xlabel('频率 / MHz'); ylabel('功率 / dB');
title('I路实信号FFT（对称）'); grid on;

% Q路FFT（实信号，对称）
subplot(2, 3, 5);
Y_Q = fftshift(fft(Q_clean(1:min(Nfft, length(Q_clean)))));
plot(f/1e6, 10*log10(abs(Y_Q).^2 + eps), 'Color', [0.8 0.3 0.2], 'LineWidth', 0.8);
xlim([-30 30]); ylim([-80 40]);
xlabel('频率 / MHz'); ylabel('功率 / dB');
title('Q路实信号FFT（对称）'); grid on;

% 复信号FFT（单边，反映真实频率）
subplot(2, 3, 6);
Y_complex = fftshift(fft(allData_clean(1:min(Nfft, length(allData_clean)))));
plot(f/1e6, 10*log10(abs(Y_complex).^2 + eps), 'Color', [0.3 0.6 0.3], 'LineWidth', 0.8);
xlim([-30 30]); ylim([-80 40]);
xlabel('频率 / MHz'); ylabel('功率 / dB');
title('复信号FFT（单边，真实频率）'); grid on;

sgtitle(sprintf('单帧数据 | 帧长=%d样本 (%.3f ms)', ...
    frameSamples, frameSamples/params.sampleRate*1000), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% ========== 图2/3：多片段 0.05s STFT（在 2s 内均匀抽取） ==========
segMs = params.segDuration * 1000;
segSamples = round(params.sampleRate * params.segDuration);
maxStartIdx = length(allData_clean) - segSamples + 1;
segStartIdx = round(linspace(1, maxStartIdx, params.numSegPlots));

figPaths = {fullfile(stateDir, sprintf('%d_fig1_frame.png', recordIndex))};
saveRecordingFigure(fig1, figPaths{1});

for segPlotIdx = 1:params.numSegPlots
    startIdx = segStartIdx(segPlotIdx);
    tOffsetSec = (startIdx - 1) / params.sampleRate;
    complexSeg = allData_clean(startIdx:startIdx + segSamples - 1);
    [fig2, fig3] = plotSegmentFigures(complexSeg, params, segMs, tOffsetSec);

    fig2Path = fullfile(stateDir, sprintf('%d_fig2_stft_%.2fs.png', recordIndex, tOffsetSec));
    fig3Path = fullfile(stateDir, sprintf('%d_fig3_spectrum_%.2fs.png', recordIndex, tOffsetSec));
    saveRecordingFigure(fig2, fig2Path);
    saveRecordingFigure(fig3, fig3Path);
    figPaths{end+1} = fig2Path; %#ok<AGROW>
    figPaths{end+1} = fig3Path; %#ok<AGROW>
end

fprintf('[OK] 可视化完成，共 %d 个片段，图片已保存:\n', params.numSegPlots);
for k = 1:numel(figPaths)
    fprintf('  %s\n', figPaths{k});
end

function [fig2, fig3] = plotSegmentFigures(complexSeg, params, segMs, tOffsetSec)
window = params.stftWindow;
noverlap = window - params.stftHop;
nfft = params.stftNfft;

[s_complex, f_stft, t_stft] = spectrogram(complexSeg, hamming(window), noverlap, nfft, params.sampleRate, 'centered');
s_mag = abs(s_complex);
s_mag_db = 20*log10(s_mag + eps);
f_mhz = f_stft / 1e6;
t_ms = t_stft * 1000;
caxisRange = [max(s_mag_db(:))-80, max(s_mag_db(:))];

fig2 = figure('Name', sprintf('STFT @ %.2fs', tOffsetSec), 'Position', [50 50 1600 800], 'Color', 'w');

subplot(2, 2, 1);
imagesc(t_ms, f_mhz, s_mag_db);
axis xy; colormap(jet); colorbar;
xlabel('时间 (ms)'); ylabel('频率 (MHz)');
title('复信号幅度STFT |I+jQ|（单边频谱）');
ylim([-30 30]); xlim([0 segMs]); caxis(caxisRange);

subplot(2, 2, 2);
[s_I_wrong, ~, ~] = spectrogram(double(real(complexSeg)), hamming(window), noverlap, nfft, params.sampleRate, 'centered');
s_I_mag_db = 20*log10(abs(s_I_wrong) + eps);
imagesc(t_ms, f_mhz, s_I_mag_db);
axis xy; colormap(jet); colorbar;
xlabel('时间 (ms)'); ylabel('频率 (MHz)');
title('【错误示范】I路实信号STFT（对称）');
ylim([-30 30]); xlim([0 segMs]); caxis(caxisRange);

subplot(2, 2, 3);
decim = 20;
complexDown = complexSeg(1:decim:end);
fs_down = params.sampleRate / decim;
instPhase = unwrap(angle(complexDown));
instFreq = diff(instPhase) / (2*pi) * fs_down;
t_inst = (0:length(instFreq)-1) / fs_down * 1000;
instFreq_filt = medfilt1(instFreq, 5);
plot(t_inst, instFreq_filt/1e6, '.', 'MarkerSize', 1, 'Color', [0.2 0.4 0.8]);
xlabel('时间 (ms)'); ylabel('瞬时频率 (MHz)');
title('瞬时频率轨迹（降采样+中值滤波）');
grid on; ylim([-30 30]); xlim([0 segMs]);

subplot(2, 2, 4);
plot(t_inst, instPhase(1:length(t_inst)), 'Color', [0.3 0.6 0.3], 'LineWidth', 0.5);
xlabel('时间 (ms)'); ylabel('相位 (rad)');
title('复信号相位轨迹（unwrap后）');
grid on; xlim([0 segMs]);

sgtitle(sprintf('%.2fs片段 @ %.2fs | 窗长%d/间隔%d | 采样率%.2f MHz', ...
    segMs/1000, tOffsetSec, window, params.stftHop, params.sampleRate/1e6), ...
    'FontSize', 14, 'FontWeight', 'bold');

fig3 = figure('Name', sprintf('频谱分解 @ %.2fs', tOffsetSec), 'Position', [50 50 1600 500], 'Color', 'w');
maxFreqOffset = params.sampleRate / 2 / 1e6;
freqOffset = linspace(-maxFreqOffset, maxFreqOffset, size(s_complex, 1));

subplot(1, 3, 1);
real_db = 20*log10(abs(real(s_complex)) + eps);
imagesc(t_ms, freqOffset, real_db);
axis xy; colormap(jet); colorbar;
xlabel('时间 (ms)'); ylabel('频率偏移 (MHz)');
title('STFT实部 |Re{Zxx}| (dB)');
ylim([-30 30]); xlim([0 segMs]); caxis(caxisRange);

subplot(1, 3, 2);
imag_db = 20*log10(abs(imag(s_complex)) + eps);
imagesc(t_ms, freqOffset, imag_db);
axis xy; colormap(jet); colorbar;
xlabel('时间 (ms)'); ylabel('频率偏移 (MHz)');
title('STFT虚部 |Im{Zxx}| (dB)');
ylim([-30 30]); xlim([0 segMs]); caxis(caxisRange);

subplot(1, 3, 3);
imagesc(t_ms, freqOffset, s_mag_db);
axis xy; colormap(jet); colorbar;
xlabel('时间 (ms)'); ylabel('频率偏移 (MHz)');
title('复数幅度 |Zxx| (dB) —— 与Python一致');
ylim([-30 30]); xlim([0 segMs]); caxis(caxisRange);

sgtitle(sprintf('复频谱分解 @ %.2fs | 实部/虚部/幅度 (dB)', tOffsetSec), ...
    'FontSize', 14, 'FontWeight', 'bold');
end

function printSignalQualityReport(iqData)
amp = abs(iqData);
ampMax = max(amp);
step = max(1, floor(numel(amp) / 100000));
ampSample = amp(1:step:end);
sortedAmp = sort(ampSample);
ampP99 = sortedAmp(max(1, round(0.99 * numel(sortedAmp))));
fprintf('信号质量: |IQ| max=%.4f  p99≈%.4f  mean=%.6f\n', ampMax, ampP99, mean(amp));
if ampMax >= 1.0
    fprintf('[警告] 可能过载/削顶 (|IQ|>=1.0)，建议降低增益后重采\n');
elseif ampMax < 0.1
    fprintf('[提示] 信号偏弱 (|IQ|max<0.1)，可适当提高增益\n');
elseif ampMax >= 0.8
    fprintf('[提示] 接近满幅 (|IQ|max>=0.8)，若重采可考虑略降增益\n');
else
    fprintf('[OK] 幅度正常 (建议范围 0.3~0.7)\n');
end
end

function releaseStaleSDRuReceivers()
% 释放工作区里残留的 comm.SDRuReceiver，避免 "radio is busy"
vars = evalin('base', 'whos');
for k = 1:numel(vars)
    if ~strcmp(vars(k).class, 'comm.SDRuReceiver')
        continue;
    end
    try
        obj = evalin('base', vars(k).name);
        release(obj);
    catch
    end
    evalin('base', sprintf('clear %s', vars(k).name));
end
end

function saveRecordingFigure(figHandle, filepath)
    try
        exportgraphics(figHandle, filepath, 'Resolution', 150);
    catch
        saveas(figHandle, filepath);
    end
end
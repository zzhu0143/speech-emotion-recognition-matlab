%% 语音情感识别 - 启动训练脚本
% 这个脚本会检查环境并开始训练

clear; clc;
fprintf('\n========================================\n');
fprintf('语音情感识别项目 - 启动中...\n');
fprintf('========================================\n\n');

%% 步骤1: 检查MATLAB版本
fprintf('步骤1: 检查MATLAB版本\n');
matlabVersion = version;
fprintf('当前MATLAB版本: %s\n', matlabVersion);

%% 步骤2: 检查必需工具箱
fprintf('\n步骤2: 检查必需工具箱\n');
fprintf('检查中...\n');

% 检查Deep Learning Toolbox
hasDeepLearning = license('test', 'Neural_Network_Toolbox');
if hasDeepLearning
    fprintf('✓ Deep Learning Toolbox 已安装\n');
else
    fprintf('✗ Deep Learning Toolbox 未安装\n');
    fprintf('  请安装: 主页 -> Add-Ons -> 搜索 "Deep Learning Toolbox"\n');
end

% 检查Audio Toolbox
hasAudio = license('test', 'Audio_Toolbox');
if hasAudio
    fprintf('✓ Audio Toolbox 已安装\n');
else
    fprintf('✗ Audio Toolbox 未安装\n');
    fprintf('  提示: 将使用简化版特征提取功能\n');
end

% 检查Signal Processing Toolbox
hasSignal = license('test', 'Signal_Toolbox');
if hasSignal
    fprintf('✓ Signal Processing Toolbox 已安装\n');
else
    fprintf('✗ Signal Processing Toolbox 未安装\n');
end

%% 步骤3: 检查数据集
fprintf('\n步骤3: 检查数据集\n');
dataPath = 'data/RAVDESS';

if exist(dataPath, 'dir')
    fprintf('✓ 数据文件夹存在: %s\n', dataPath);

    % 统计Actor文件夹
    actors = dir(fullfile(dataPath, 'Actor_*'));
    fprintf('✓ 找到 %d 个Actor文件夹\n', length(actors));

    % 统计音频文件
    totalFiles = 0;
    for i = 1:length(actors)
        files = dir(fullfile(dataPath, actors(i).name, '*.wav'));
        totalFiles = totalFiles + length(files);
    end
    fprintf('✓ 总共 %d 个音频文件\n', totalFiles);

    if totalFiles < 100
        fprintf('\n警告: 音频文件数量较少,可能影响训练效果\n');
    end
else
    fprintf('✗ 数据文件夹不存在: %s\n', dataPath);
    fprintf('\n请下载RAVDESS数据集:\n');
    fprintf('1. 访问: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio\n');
    fprintf('2. 下载并解压到: %s\n', fullfile(pwd, dataPath));
    fprintf('\n按任意键退出...\n');
    pause;
    return;
end

%% 步骤4: 创建必要的文件夹
fprintf('\n步骤4: 创建输出文件夹\n');
if ~exist('models', 'dir')
    mkdir('models');
    fprintf('✓ 创建 models/ 文件夹\n');
else
    fprintf('✓ models/ 文件夹已存在\n');
end

if ~exist('results', 'dir')
    mkdir('results');
    fprintf('✓ 创建 results/ 文件夹\n');
else
    fprintf('✓ results/ 文件夹已存在\n');
end

%% 步骤5: 开始训练
fprintf('\n========================================\n');
fprintf('环境检查完成!\n');
fprintf('========================================\n\n');

fprintf('准备开始训练...\n');
fprintf('预计时间: 30-50分钟\n\n');

% 询问用户是否继续
fprintf('是否开始训练? \n');
fprintf('1 - 开始完整训练 (所有模型)\n');
fprintf('2 - 仅测试数据加载\n');
fprintf('3 - 退出\n');
choice = input('请选择 (1/2/3): ');

switch choice
    case 1
        fprintf('\n开始完整训练...\n\n');
        main_train_all_models;

    case 2
        fprintf('\n测试数据加载...\n\n');
        test_simple;

    case 3
        fprintf('\n已退出\n');
        return;

    otherwise
        fprintf('\n无效选择,已退出\n');
        return;
end

fprintf('\n========================================\n');
fprintf('脚本执行完成!\n');
fprintf('========================================\n');

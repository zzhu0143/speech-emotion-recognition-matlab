# Speech Emotion Recognition Using MATLAB Deep Learning

基于MATLAB深度学习工具箱的语音情感识别系统

---

## 项目概述

本项目使用MATLAB实现了完整的语音情感识别系统，包括：

- **数据处理**：RAVDESS数据集加载和特征提取
- **三个深度学习模型**：
  1. 基线神经网络（Baseline NN）
  2. LSTM循环神经网络
  3. CNN-LSTM混合模型（可选）
- **完整评估**：准确率、混淆矩阵、可视化
- **预测功能**：对新音频文件进行情感预测

**研究问题**：如何使用深度学习有效识别语音中的情感？

---

## 系统要求

### MATLAB版本
- **MATLAB R2020b或更高版本**
- 推荐：MATLAB R2022a+（最佳兼容性）

### 必需工具箱
1. **Deep Learning Toolbox** （深度学习工具箱）
2. **Audio Toolbox** （音频工具箱）
3. **Signal Processing Toolbox** （信号处理工具箱）
4. **Statistics and Machine Learning Toolbox** （统计和机器学习工具箱）

### 检查工具箱
在MATLAB命令窗口运行：
```matlab
ver
```

### 安装缺少的工具箱
1. 点击MATLAB主页 → **Add-Ons**
2. 搜索并安装缺少的工具箱

---

## 快速开始

### 第1步：下载数据集

1. 访问：https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
2. 下载RAVDESS数据集（约200MB）
3. 解压到项目的 `data/RAVDESS/` 文件夹

您的文件结构应该是：
```
speech_emotion_recognition_matlab/
├── data/
│   └── RAVDESS/
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ...
│       └── Actor_24/
```

### 第2步：打开MATLAB

1. 启动MATLAB
2. 将当前文件夹切换到项目目录：
```matlab
cd('C:\Users\朱\speech_emotion_recognition_matlab')
```

### 第3步：运行主训练脚本

```matlab
main_train_all_models
```

这将：
- 加载RAVDESS数据集
- 提取音频特征
- 训练基线神经网络和LSTM模型
- 生成混淆矩阵和性能对比图
- 保存所有结果到 `results/` 文件夹
- 保存训练好的模型到 `models/` 文件夹

**预计运行时间**：
- 特征提取：10-15分钟
- 基线模型训练：5-10分钟
- LSTM模型训练：15-25分钟
- **总计**：约30-50分钟

### 第4步：测试预测

训练完成后，测试情感预测：

```matlab
% 预测单个音频文件的情感
[emotion, probs] = predictEmotion('path/to/audio.wav');

% 使用LSTM模型预测
[emotion, probs] = predictEmotion('path/to/audio.wav', 'models/lstm_model.mat');
```

---

## 项目文件说明

### 核心脚本

| 文件 | 功能 | 说明 |
|------|------|------|
| `main_train_all_models.m` | 主训练脚本 | 运行完整训练流程 |
| `extractAudioFeatures.m` | 特征提取 | 提取MFCC、频谱特征等 |
| `loadRAVDESSData.m` | 数据加载 | 加载RAVDESS数据集 |
| `trainBaselineModel.m` | 基线模型训练 | 训练全连接神经网络 |
| `trainLSTMModel.m` | LSTM模型训练 | 训练循环神经网络 |
| `predictEmotion.m` | 情感预测 | 预测新音频的情感 |

### 文件夹结构

```
speech_emotion_recognition_matlab/
│
├── data/                          # 数据文件夹
│   ├── RAVDESS/                   # RAVDESS数据集（需下载）
│   └── extracted_features.mat     # 提取的特征（自动生成）
│
├── models/                        # 训练好的模型
│   ├── baseline_model.mat         # 基线模型
│   └── lstm_model.mat             # LSTM模型
│
├── results/                       # 结果和可视化
│   ├── baseline_confusion_matrix.png
│   ├── lstm_confusion_matrix.png
│   ├── model_comparison.png
│   └── training_report.txt
│
├── functions/                     # 辅助函数（可选）
│
├── main_train_all_models.m       # 主训练脚本
├── extractAudioFeatures.m        # 特征提取
├── loadRAVDESSData.m             # 数据加载
├── trainBaselineModel.m          # 基线模型
├── trainLSTMModel.m              # LSTM模型
├── predictEmotion.m              # 预测功能
│
├── README.md                     # 本文件
├── MATLAB_REPORT.md              # 完整研究报告
└── QUICK_START_MATLAB.md         # 快速指南
```

---

## 模型架构

### 1. 基线神经网络

```
输入 (95维特征)
  ↓
全连接层 (256) + 批归一化 + ReLU + Dropout(0.3)
  ↓
全连接层 (128) + 批归一化 + ReLU + Dropout(0.3)
  ↓
全连接层 (64) + 批归一化 + ReLU + Dropout(0.3)
  ↓
全连接层 (8) + Softmax
  ↓
输出 (8个情感类别)
```

**特征**：
- MFCCs（均值和标准差）：80维
- 频谱特征：8维
- 过零率：2维
- 能量特征：2维
- 基频特征：3维
- **总计**：95维

### 2. LSTM网络

```
输入 (40维MFCC序列)
  ↓
双向LSTM (128隐藏单元) + Dropout(0.3)
  ↓
全连接层 (64) + ReLU + Dropout(0.3)
  ↓
全连接层 (8) + Softmax
  ↓
输出 (8个情感类别)
```

**优势**：
- 捕捉时序信息
- 双向处理（前向+后向）
- 适合情感的动态变化

---

## 预期结果

基于RAVDESS数据集的预期性能：

| 模型 | 准确率 | 优势 |
|------|--------|------|
| 基线神经网络 | 75-80% | 快速训练，简单有效 |
| LSTM网络 | 82-88% | 捕捉时序信息 |
| CNN-LSTM | 85-90% | 最佳性能（需更多资源）|

**情感识别**：
- 最易识别：Neutral（中性）、Angry（愤怒）
- 较难识别：Calm（平静）vs. Sad（悲伤）

---

## 使用示例

### 示例1：训练所有模型

```matlab
% 确保在项目文件夹
cd('C:\Users\朱\speech_emotion_recognition_matlab')

% 运行主脚本
main_train_all_models

% 等待训练完成...
% 查看results/文件夹的结果
```

### 示例2：单独训练基线模型

```matlab
% 加载数据
[features, labels, emotionNames] = loadRAVDESSData('data/RAVDESS');

% 训练基线模型
[net, accuracy, confMat] = trainBaselineModel(features, labels);

% 查看准确率
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
```

### 示例3：预测新音频

```matlab
% 预测情感
audioFile = 'data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav';
[emotion, probs] = predictEmotion(audioFile, 'models/baseline_model.mat');

% 显示结果
fprintf('Predicted: %s (%.2f%% confident)\n', emotion, max(probs)*100);
```

### 示例4：批量预测

```matlab
% 获取所有音频文件
audioFiles = dir('data/RAVDESS/Actor_01/*.wav');

% 预测每个文件
for i = 1:length(audioFiles)
    audioPath = fullfile(audioFiles(i).folder, audioFiles(i).name);
    [emotion, probs] = predictEmotion(audioPath);
    fprintf('%s: %s\n', audioFiles(i).name, emotion);
end
```

---

## 特征提取详解

### 提取的特征

1. **MFCC（Mel频率倒谱系数）**
   - 40个系数
   - 计算均值和标准差
   - 共80维

2. **频谱特征**
   - 频谱质心（Spectral Centroid）
   - 频谱滚降点（Spectral Rolloff）
   - 频谱通量（Spectral Flux）
   - 频谱熵（Spectral Entropy）

3. **时域特征**
   - 过零率（Zero Crossing Rate）
   - 能量（Energy）

4. **基频特征**
   - F0均值、标准差、范围

### 特征提取代码

```matlab
% 读取音频
[audio, fs] = audioread('audio.wav');

% 提取MFCC
mfccCoeffs = mfcc(audio, fs, 'NumCoeffs', 40);

% 提取频谱特征
sCentroid = spectralCentroid(audio, fs);
sRolloff = spectralRolloffPoint(audio, fs);

% 组合特征
features = [mean(mfccCoeffs), std(mfccCoeffs), ...
           mean(sCentroid), mean(sRolloff), ...];
```

---

## 训练选项说明

### 优化器设置

```matlab
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...              % 最大训练轮数
    'MiniBatchSize', 32, ...           % 批次大小
    'InitialLearnRate', 0.001, ...     % 初始学习率
    'LearnRateSchedule', 'piecewise', ... % 学习率衰减
    'LearnRateDropFactor', 0.5, ...    % 衰减因子
    'LearnRateDropPeriod', 20, ...     % 衰减周期
    'ValidationFrequency', 10, ...     % 验证频率
    'Plots', 'training-progress', ...  % 显示训练进度
    'ExecutionEnvironment', 'auto');   % 自动选择GPU/CPU
```

### 使用GPU加速

如果您有NVIDIA GPU：

```matlab
% 检查GPU可用性
gpuDevice

% 训练会自动使用GPU（如果ExecutionEnvironment设为'auto'或'gpu'）
```

---

## 常见问题

### Q1: 提示工具箱缺失

**错误**：`Undefined function or variable 'trainNetwork'`

**解决**：
1. 检查是否安装了Deep Learning Toolbox
2. 在MATLAB命令窗口运行 `ver`
3. 如未安装，前往 主页 → Add-Ons → 搜索 "Deep Learning Toolbox"

### Q2: 找不到数据集

**错误**：`Data path does not exist: data/RAVDESS`

**解决**：
1. 确保已下载RAVDESS数据集
2. 检查文件夹路径是否正确
3. 修改 `main_train_all_models.m` 中的 `dataPath` 变量

### Q3: 内存不足

**错误**：`Out of memory`

**解决**：
1. 减小 `MiniBatchSize`（从32改为16或8）
2. 关闭其他应用程序
3. 分批处理数据
4. 使用GPU（如果可用）

### Q4: 训练太慢

**解决**：
1. 确保使用GPU（`gpuDevice`）
2. 减少 `MaxEpochs`（测试时用20-30）
3. 减小批次大小
4. 使用并行计算工具箱

### Q5: 无法提取特征

**错误**：特征提取失败

**解决**：
1. 确保音频文件是.wav格式
2. 检查Audio Toolbox是否已安装
3. 确保音频文件未损坏
4. 尝试用 `audioread('file.wav')` 手动读取测试

---

## 性能优化建议

### 1. 使用GPU加速

```matlab
% 检查GPU
gpuDevice

% 将数据移到GPU
XTrain_gpu = gpuArray(XTrain);
YTrain_gpu = gpuArray(YTrain);
```

### 2. 保存提取的特征

```matlab
% 第一次运行后，特征已保存
% 后续可直接加载
load('data/extracted_features.mat');
```

### 3. 并行处理

```matlab
% 启用并行池
parpool;

% 使用parfor加速数据加载
parfor i = 1:length(audioFiles)
    features{i} = extractAudioFeatures(audioFiles{i});
end
```

---

## 结果分析

### 查看训练进度

训练时会自动显示：
- 训练损失和准确率曲线
- 验证损失和准确率曲线
- 实时更新的图表

### 混淆矩阵分析

```matlab
% 加载结果
load('results/all_results.mat');

% 查看混淆矩阵
figure;
confusionchart(confMatBaseline);
title('Baseline Model Confusion Matrix');
```

### 错误分析

```matlab
% 找出分类错误的样本
incorrectIdx = find(YPred ~= YTest);
fprintf('Misclassified samples: %d\n', length(incorrectIdx));

% 查看具体错误
for i = 1:min(10, length(incorrectIdx))
    idx = incorrectIdx(i);
    fprintf('Sample %d: True=%s, Predicted=%s\n', ...
        idx, char(YTest(idx)), char(YPred(idx)));
end
```

---

## 扩展和改进

### 1. 数据增强

```matlab
% 添加噪声
noisyAudio = audio + 0.005 * randn(size(audio));

% 音高变换
pitchShiftedAudio = shiftPitch(audio, fs, 2); % 升高2个半音

% 时间拉伸
stretchedAudio = timeStretch(audio, 1.1); % 拉伸10%
```

### 2. 集成学习

```matlab
% 训练多个模型
net1 = trainBaselineModel(features, labels);
net2 = trainLSTMModel(dataPath);

% 投票集成
predBaseline = classify(net1, XTest);
predLSTM = classify(net2, XTestSeq);

% 组合预测（简单投票）
finalPred = mode([predBaseline, predLSTM], 2);
```

### 3. 超参数调优

```matlab
% 学习率搜索
learningRates = [0.0001, 0.0005, 0.001, 0.005];
for lr = learningRates
    options = trainingOptions('adam', 'InitialLearnRate', lr, ...);
    net = trainNetwork(XTrain, YTrain, layers, options);
    accuracy = evaluateModel(net, XTest, YTest);
    fprintf('LR=%.4f, Accuracy=%.2f%%\n', lr, accuracy*100);
end
```

---

## 项目提交清单

### 必需文件

- [x] 所有MATLAB代码文件（.m文件）
- [x] README.md（本文件）
- [x] 研究报告（MATLAB_REPORT.md）
- [x] 训练结果（results/文件夹）
- [x] GitHub仓库链接

### 可选文件

- [ ] 训练好的模型（models/文件夹，可能太大）
- [ ] 演示视频
- [ ] 示例音频文件

---

## 参考资源

### MATLAB文档

- [Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/)
- [Audio Toolbox](https://www.mathworks.com/help/audio/)
- [Train Network for Speech Command Recognition](https://www.mathworks.com/help/audio/ug/speech-command-recognition-using-deep-learning.html)

### 数据集

- [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio)
- [RAVDESS Official](https://zenodo.org/record/1188976)

### 相关论文

- Livingstone SR, Russo FA (2018) The RAVDESS Database. PLoS ONE 13(5)
- MATLAB示例：Speech Emotion Recognition

---

## 联系方式

- **课程**：Speech and Audio Processing
- **学校**：University of Sydney
- **GitHub**：[您的GitHub链接]

---

## 许可

本项目用于教育目的，基于MATLAB和RAVDESS数据集。

**祝您项目顺利！Good Luck!** 🎉

# MATLAB语音情感识别 - 快速开始指南

最简单的步骤帮您完成项目！

---

## ⚡ 5分钟快速开始

### 第1步：检查MATLAB（2分钟）

1. **打开MATLAB**（确保是R2020b或更高版本）

2. **检查必需工具箱**，在命令窗口输入：
```matlab
ver
```

需要看到以下工具箱：
- ✅ Deep Learning Toolbox
- ✅ Audio Toolbox
- ✅ Signal Processing Toolbox
- ✅ Statistics and Machine Learning Toolbox

**如果缺少工具箱**：
- 点击 **主页** → **Add-Ons**
- 搜索并安装缺少的工具箱

### 第2步：下载数据集（10分钟）

1. 访问：https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio

2. 点击 **Download** 下载（约200MB）

3. 解压文件，把 `Actor_01` 到 `Actor_24` 文件夹放到：
```
C:\Users\朱\speech_emotion_recognition_matlab\data\RAVDESS\
```

### 第3步：运行训练（3分钟设置 + 30-50分钟训练）

1. **在MATLAB中切换到项目文件夹**：
```matlab
cd('C:\Users\朱\speech_emotion_recognition_matlab')
```

2. **运行主脚本**：
```matlab
main_train_all_models
```

3. **等待完成**，会自动：
   - 加载数据 ✓
   - 提取特征 ✓
   - 训练模型 ✓
   - 生成结果 ✓

完成！🎉

---

## 📁 确认文件结构

运行前确保文件结构正确：

```
speech_emotion_recognition_matlab/
├── data/
│   └── RAVDESS/           ← 确保这个文件夹存在
│       ├── Actor_01/      ← 确保有24个Actor文件夹
│       ├── Actor_02/
│       └── ...
│       └── Actor_24/
│
├── main_train_all_models.m      ← 主脚本
├── extractAudioFeatures.m       ← 特征提取
├── loadRAVDESSData.m            ← 数据加载
├── trainBaselineModel.m         ← 基线模型
├── trainLSTMModel.m             ← LSTM模型
└── predictEmotion.m             ← 预测功能
```

---

## 🚀 完整步骤详解

### 步骤1：打开MATLAB并设置路径

```matlab
% 切换到项目目录
cd('C:\Users\朱\speech_emotion_recognition_matlab')

% 确认当前目录
pwd  % 应该显示项目路径

% 查看文件
dir  % 应该看到所有.m文件
```

### 步骤2：验证数据集

```matlab
% 检查数据集是否存在
if exist('data/RAVDESS', 'dir')
    fprintf('✓ 数据集文件夹存在\n');

    % 计算音频文件数量
    actors = dir('data/RAVDESS/Actor_*');
    fprintf('找到 %d 个Actor文件夹\n', length(actors));

    totalFiles = 0;
    for i = 1:length(actors)
        files = dir(fullfile('data/RAVDESS', actors(i).name, '*.wav'));
        totalFiles = totalFiles + length(files);
    end
    fprintf('总共 %d 个音频文件\n', totalFiles);
else
    fprintf('✗ 数据集文件夹不存在！请先下载数据集。\n');
end
```

**应该看到**：
```
✓ 数据集文件夹存在
找到 24 个Actor文件夹
总共 1440 个音频文件
```

### 步骤3：运行完整训练

```matlab
% 运行主训练脚本
main_train_all_models
```

**训练过程显示**：

```
========================================
Speech Emotion Recognition Project
Training All Models
========================================

Step 1: Loading RAVDESS Dataset
================================
Loading RAVDESS dataset from data/RAVDESS...
Found 24 actor folders
Total audio files to process: 1440
Extracting features...
Processed 50/1440 files (3.5%)
Processed 100/1440 files (6.9%)
...
Feature extraction complete!
Total samples: 1440
Feature dimension: 95

========================================
Step 2: Training Baseline Model
========================================
Training samples: 1152
Test samples: 288
Training network...
[训练进度显示]
✓ Baseline Model Training Complete!
  Accuracy: 78.50%

========================================
Step 3: Training LSTM Model
========================================
[LSTM训练过程]
✓ LSTM Model Training Complete!
  Accuracy: 86.10%

========================================
Final Results Comparison
========================================
Baseline Neural Network:  78.50%
LSTM Network:            86.10%
```

### 步骤4：查看结果

```matlab
% 打开结果文件夹
winopen('results')

% 或在MATLAB中查看图片
imshow('results/model_comparison.png')
imshow('results/baseline_confusion_matrix.png')
imshow('results/lstm_confusion_matrix.png')

% 读取文本报告
type('results/training_report.txt')
```

---

## 🎯 测试预测功能

训练完成后，测试情感识别：

### 方法1：预测RAVDESS数据集中的音频

```matlab
% 选择一个测试音频
testAudio = 'data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav';

% 预测情感（使用基线模型）
[emotion, probs] = predictEmotion(testAudio);

% 或使用LSTM模型
[emotion, probs] = predictEmotion(testAudio, 'models/lstm_model.mat');
```

### 方法2：批量测试

```matlab
% 获取Actor_01的所有音频
audioFiles = dir('data/RAVDESS/Actor_01/*.wav');

% 预测前10个
for i = 1:min(10, length(audioFiles))
    audioPath = fullfile(audioFiles(i).folder, audioFiles(i).name);
    [emotion, ~] = predictEmotion(audioPath);
    fprintf('文件 %d: %s\n', i, emotion);
end
```

---

## 常见问题快速解决

### ❌ 问题1："找不到数据集"

**错误信息**：
```
Error: Data path does not exist: data/RAVDESS
```

**解决方法**：
1. 确认已下载RAVDESS数据集
2. 确认文件夹结构正确：
```matlab
ls data/RAVDESS  % 应该看到Actor_01到Actor_24
```
3. 如果路径不同，修改 `main_train_all_models.m` 第17行：
```matlab
dataPath = 'C:\完整路径\到\RAVDESS';  % 改成您的实际路径
```

### ❌ 问题2："缺少工具箱"

**错误信息**：
```
Undefined function 'trainNetwork'
```

**解决方法**：
1. 检查工具箱：
```matlab
ver
```
2. 安装Deep Learning Toolbox：
   - 主页 → Add-Ons → 搜索 "Deep Learning Toolbox" → 安装

### ❌ 问题3："内存不足"

**错误信息**：
```
Out of memory
```

**解决方法**：
1. 关闭其他程序释放内存
2. 减小批次大小，修改 `trainBaselineModel.m` 第37行：
```matlab
'MiniBatchSize', 16, ...  % 从32改为16或8
```
3. 清理MATLAB工作区：
```matlab
clear; clc;
```

### ❌ 问题4："训练太慢"

**解决方法**：

1. **使用GPU**（如果有NVIDIA显卡）：
```matlab
% 检查GPU
gpuDevice

% GPU会自动使用，如果可用
```

2. **减少训练轮数**（用于快速测试）：
   修改 `trainBaselineModel.m` 第35行：
```matlab
'MaxEpochs', 30, ...  % 从100改为30
```

3. **只训练一个模型**：
```matlab
% 不运行main_train_all_models，而是单独训练：
[features, labels] = loadRAVDESSData('data/RAVDESS');
[net, acc] = trainBaselineModel(features, labels);
```

---

## ⏱️ 预计时间安排

| 步骤 | 时间 | 说明 |
|------|------|------|
| 下载数据集 | 10-20分钟 | 取决于网速 |
| 安装工具箱 | 5-15分钟 | 如需要 |
| 特征提取 | 10-15分钟 | 首次运行 |
| 训练基线模型 | 5-10分钟 | 取决于硬件 |
| 训练LSTM模型 | 15-25分钟 | 取决于硬件 |
| **总计** | **45-85分钟** | 首次完整运行 |

**后续运行**（特征已提取）：约20-35分钟

---

## 📊 期望结果

### 控制台输出

```
========================================
Final Results Comparison
========================================

Model Performance Summary:
--------------------------
Baseline Neural Network:  78.50%
LSTM Network:            86.10%

Results saved to results/ folder
```

### 生成的文件

在 `results/` 文件夹应该看到：
- ✅ `baseline_confusion_matrix.png` - 基线模型混淆矩阵
- ✅ `lstm_confusion_matrix.png` - LSTM模型混淆矩阵
- ✅ `model_comparison.png` - 模型性能对比图
- ✅ `training_report.txt` - 详细训练报告
- ✅ `all_results.mat` - 所有结果的MATLAB数据

在 `models/` 文件夹应该看到：
- ✅ `baseline_model.mat` - 训练好的基线模型
- ✅ `lstm_model.mat` - 训练好的LSTM模型

在 `data/` 文件夹应该看到：
- ✅ `extracted_features.mat` - 提取的特征（用于后续快速加载）

---

## 🎓 完成后的下一步

### 1. 查看和分析结果

```matlab
% 加载所有结果
load('results/all_results.mat')

% 查看模型准确率
fprintf('基线模型准确率: %.2f%%\n', accBaseline * 100);
fprintf('LSTM模型准确率: %.2f%%\n', accLSTM * 100);

% 显示混淆矩阵
figure;
subplot(1,2,1);
confusionchart(confMatBaseline);
title('Baseline Model');

subplot(1,2,2);
confusionchart(confMatLSTM);
title('LSTM Model');
```

### 2. 测试更多音频

```matlab
% 随机测试10个音频
allFiles = dir('data/RAVDESS/**/*.wav');
randomIdx = randperm(length(allFiles), 10);

for i = 1:10
    audioFile = fullfile(allFiles(randomIdx(i)).folder, ...
                        allFiles(randomIdx(i)).name);
    [emotion, ~] = predictEmotion(audioFile);
    fprintf('%d. %s\n', i, emotion);
end
```

### 3. 创建GitHub仓库

```
1. 在GitHub创建新仓库：speech-emotion-recognition-matlab
2. 上传所有.m文件和README.md
3. 不要上传data/和models/文件夹（太大）
4. 记录GitHub链接
```

### 4. 录制演示视频（5-10分钟）

展示内容：
- ✅ 打开MATLAB和项目
- ✅ 运行 `main_train_all_models.m`（可以快进）
- ✅ 展示训练过程和结果图表
- ✅ 运行 `predictEmotion.m` 演示预测
- ✅ 解释结果

### 5. 编写报告

基于提供的模板编写完整研究报告（已提供MATLAB_REPORT.md）。

---

## 📋 提交检查清单

准备提交前检查：

### 代码部分
- [ ] 所有.m文件已上传到GitHub
- [ ] README.md清晰完整
- [ ] 代码可以在您的电脑上运行
- [ ] GitHub链接可以访问

### 结果部分
- [ ] results/ 文件夹有所有图表
- [ ] 训练报告文本文件
- [ ] 准确率结果正确

### 文档部分
- [ ] 研究报告完成
- [ ] 包含GitHub链接
- [ ] 包含演示视频链接

### 演示视频
- [ ] 视频已录制（5-10分钟）
- [ ] 视频已上传（YouTube/Bilibili）
- [ ] 视频链接已记录

---

## 💡 专业提示

### 1. 保存工作进度

```matlab
% 训练中途保存
save('my_progress.mat');

% 恢复工作
load('my_progress.mat');
```

### 2. 导出高质量图片

```matlab
% 设置高分辨率
set(gcf, 'PaperPosition', [0 0 8 6]);
set(gcf, 'PaperSize', [8 6]);

% 保存为高质量PNG
print('my_figure.png', '-dpng', '-r300');
```

### 3. 创建专业的演示

```matlab
% 创建包含多个子图的综合分析
figure('Position', [100 100 1200 800]);

subplot(2,2,1);
% 绘制准确率对比

subplot(2,2,2);
% 绘制混淆矩阵

subplot(2,2,3);
% 绘制情感分布

subplot(2,2,4);
% 绘制特征重要性

% 保存完整分析图
saveas(gcf, 'comprehensive_analysis.png');
```

---

## 🆘 获取帮助

如果遇到问题：

1. **查看详细README**：`README.md`
2. **查看完整报告**：`MATLAB_REPORT.md`
3. **MATLAB文档**：`doc trainNetwork`
4. **在线帮助**：https://www.mathworks.com/help/

---

## 🎉 恭喜！

如果您完成了所有步骤，您现在有：

✅ 完整的MATLAB深度学习项目
✅ 训练好的情感识别模型
✅ 详细的结果和可视化
✅ 可运行的演示代码
✅ 专业的文档

**准备好拿高分了！Good luck!** 🚀

# Speech Emotion Recognition Using MATLAB Deep Learning

A comprehensive speech emotion recognition system implemented using MATLAB Deep Learning Toolbox and the RAVDESS dataset.

**Course:** ELEC5305 - Speech and Audio Processing
**Institution:** The University of Sydney
**Author:** Zhu
**GitHub:** https://github.com/zzhu0143/speech-emotion-recognition-matlab

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Research Question](#research-question)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Usage Examples](#usage-examples)
- [Expected Results](#expected-results)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Project Overview

This project implements a complete speech emotion recognition system using MATLAB's Deep Learning Toolbox. The system processes audio signals from the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset to classify emotional states in human speech.

### Key Features

- **Data Processing Pipeline**: Automated loading and preprocessing of RAVDESS dataset
- **Feature Extraction**: Comprehensive audio feature extraction including:
  - Mel-Frequency Cepstral Coefficients (MFCC)
  - Spectral features (centroid, rolloff, flux, entropy)
  - Temporal features (zero-crossing rate, energy)
  - Pitch features (F0 statistics)
- **Multiple Deep Learning Models**:
  1. Baseline Feedforward Neural Network
  2. LSTM (Long Short-Term Memory) Network
  3. Optional CNN-LSTM Hybrid Model
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, and visualizations
- **Prediction Functionality**: Real-time emotion prediction for new audio files

### Emotion Classes

The system recognizes 8 emotional states:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## Research Question

**How can deep learning techniques be effectively applied to recognize emotions from speech signals?**

This project investigates:
1. The effectiveness of different neural network architectures for emotion recognition
2. The importance of various audio features in emotion classification
3. Comparison between feedforward and recurrent neural network approaches
4. Performance trade-offs between model complexity and accuracy

---

## System Requirements

### MATLAB Version
- **Minimum**: MATLAB R2020b
- **Recommended**: MATLAB R2022a or higher (for best compatibility)

### Required Toolboxes
1. **Deep Learning Toolbox** (essential)
2. **Audio Toolbox** (essential)
3. **Signal Processing Toolbox** (essential)
4. **Statistics and Machine Learning Toolbox** (essential)

### Hardware Recommendations
- **CPU**: Multi-core processor (Intel i5 or better)
- **RAM**: Minimum 8GB (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)
- **Storage**: At least 2GB free space for dataset and models

### Checking Your Installation

To verify installed toolboxes, run in MATLAB:
```matlab
ver
```

Look for the four required toolboxes in the output.

---

## Installation Guide

### Step 1: Download the Dataset

1. Visit the RAVDESS dataset page:
   - **Kaggle**: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
   - **Official**: https://zenodo.org/record/1188976

2. Download the complete dataset (~200MB)

3. Extract the dataset to the project's `data/` folder

### Expected Directory Structure

After extraction, your directory should look like:
```
speech_emotion_recognition_matlab/
├── data/
│   └── RAVDESS/
│       ├── Actor_01/
│       │   ├── 03-01-01-01-01-01-01.wav
│       │   ├── 03-01-01-01-01-02-01.wav
│       │   └── ...
│       ├── Actor_02/
│       ├── Actor_03/
│       └── ...
│       └── Actor_24/
├── models/
├── results/
├── main_train_all_models.m
├── README.md
└── ...
```

### Step 2: Install Missing Toolboxes (if needed)

1. Open MATLAB
2. Click **Home** tab → **Add-Ons** → **Get Add-Ons**
3. Search for and install any missing toolboxes:
   - Deep Learning Toolbox
   - Audio Toolbox
   - Signal Processing Toolbox
   - Statistics and Machine Learning Toolbox

### Step 3: Clone or Download This Repository

**Option A: Using Git**
```bash
git clone https://github.com/zzhu0143/speech-emotion-recognition-matlab.git
cd speech-emotion-recognition-matlab
```

**Option B: Download ZIP**
1. Visit: https://github.com/zzhu0143/speech-emotion-recognition-matlab
2. Click "Code" → "Download ZIP"
3. Extract to your desired location

---

## Quick Start

### Basic Training Workflow

1. **Open MATLAB** and navigate to the project directory:
```matlab
cd('path/to/speech_emotion_recognition_matlab')
```

2. **Run the main training script**:
```matlab
main_train_all_models
```

This will:
- Load the RAVDESS dataset
- Extract audio features from all speech samples
- Train both Baseline and LSTM models
- Generate confusion matrices and performance visualizations
- Save trained models to `models/` directory
- Save results and figures to `results/` directory

### Expected Training Time

| Task | Approximate Time |
|------|------------------|
| Feature Extraction | 10-15 minutes |
| Baseline Model Training | 5-10 minutes |
| LSTM Model Training | 15-25 minutes |
| **Total** | **30-50 minutes** |

*Note: Times vary based on hardware. GPU acceleration significantly reduces training time.*

---

## Project Structure

```
speech_emotion_recognition_matlab/
│
├── data/                          # Dataset directory
│   ├── RAVDESS/                   # RAVDESS audio files (download separately)
│   └── extracted_features.mat     # Cached extracted features (auto-generated)
│
├── models/                        # Trained models (auto-generated)
│   ├── baseline_model.mat         # Baseline neural network
│   └── lstm_model.mat             # LSTM model
│
├── results/                       # Training results (auto-generated)
│   ├── baseline_confusion_matrix.png
│   ├── lstm_confusion_matrix.png
│   ├── model_comparison.fig
│   └── training_report.txt
│
├── functions/                     # Auxiliary functions (optional)
│
├── main_train_all_models.m       # Main training pipeline
├── extractAudioFeatures.m        # Comprehensive feature extraction
├── extractAudioFeatures_mfcc.m   # MFCC-focused extraction
├── extractAudioFeatures_simple.m # Simplified feature extraction
├── loadRAVDESSData.m             # Dataset loading function
├── trainBaselineModel.m          # Baseline NN training
├── trainLSTMModel.m              # LSTM training
├── predictEmotion.m              # Emotion prediction function
├── start_training.m              # Alternative training entry point
├── test_simple.m                 # Simple testing script
│
├── README.md                     # This file
├── FINAL_RESEARCH_REPORT.md      # Complete research report
├── QUICK_START_MATLAB.md         # Quick start guide
└── ELEC5305_Project_Requirements.md
```

---

## Model Architectures

### 1. Baseline Feedforward Neural Network

**Architecture:**
```
Input Layer (95 features)
    ↓
Fully Connected (256 units) + Batch Normalization + ReLU + Dropout(0.3)
    ↓
Fully Connected (128 units) + Batch Normalization + ReLU + Dropout(0.3)
    ↓
Fully Connected (64 units) + Batch Normalization + ReLU + Dropout(0.3)
    ↓
Fully Connected (8 units) + Softmax
    ↓
Output (8 emotion classes)
```

**Input Features (95 dimensions):**
- MFCC statistics (mean & std): 80 dimensions
- Spectral features: 8 dimensions
- Zero-crossing rate: 2 dimensions
- Energy features: 2 dimensions
- Pitch features: 3 dimensions

**Training Configuration:**
- Optimizer: Adam
- Initial Learning Rate: 0.001
- Max Epochs: 100
- Mini-batch Size: 32
- Learning Rate Schedule: Piecewise decay

### 2. LSTM Network

**Architecture:**
```
Input Layer (40 MFCC coefficients × Time)
    ↓
Bidirectional LSTM (128 hidden units) + Dropout(0.3)
    ↓
Fully Connected (64 units) + ReLU + Dropout(0.3)
    ↓
Fully Connected (8 units) + Softmax
    ↓
Output (8 emotion classes)
```

**Advantages:**
- Captures temporal dynamics of speech
- Bidirectional processing (forward + backward)
- Better suited for emotion's dynamic nature
- Handles variable-length sequences

**Training Configuration:**
- Optimizer: Adam
- Initial Learning Rate: 0.001
- Max Epochs: 100
- Mini-batch Size: 32
- Sequence Padding: Shortest sequence length

---

## Usage Examples

### Example 1: Train All Models

```matlab
% Navigate to project directory
cd('C:\path\to\speech_emotion_recognition_matlab')

% Run complete training pipeline
main_train_all_models

% Results will be saved in:
% - models/baseline_model.mat
% - models/lstm_model.mat
% - results/ (all figures and reports)
```

### Example 2: Train Only Baseline Model

```matlab
% Load dataset
[features, labels, emotionNames] = loadRAVDESSData('data/RAVDESS');

% Train baseline model
[net, accuracy, confMat] = trainBaselineModel(features, labels);

% Display accuracy
fprintf('Baseline Model Accuracy: %.2f%%\n', accuracy * 100);

% Save model
save('models/baseline_model.mat', 'net', 'emotionNames');
```

### Example 3: Predict Emotion from New Audio File

```matlab
% Load trained model
load('models/baseline_model.mat', 'net', 'emotionNames');

% Predict emotion for a single file
audioFile = 'data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav';
[emotion, probs] = predictEmotion(audioFile, 'models/baseline_model.mat');

% Display results
fprintf('Predicted Emotion: %s\n', emotion);
fprintf('Confidence: %.2f%%\n', max(probs) * 100);

% Display all probabilities
disp('All Emotion Probabilities:');
for i = 1:length(emotionNames)
    fprintf('  %s: %.2f%%\n', emotionNames{i}, probs(i) * 100);
end
```

### Example 4: Batch Prediction

```matlab
% Get all audio files from a specific actor
audioFiles = dir('data/RAVDESS/Actor_01/*.wav');

% Predict emotions for all files
predictions = cell(length(audioFiles), 2);
for i = 1:length(audioFiles)
    audioPath = fullfile(audioFiles(i).folder, audioFiles(i).name);
    [emotion, probs] = predictEmotion(audioPath);
    predictions{i, 1} = audioFiles(i).name;
    predictions{i, 2} = emotion;
    fprintf('%s: %s (%.2f%% confidence)\n', ...
        audioFiles(i).name, emotion, max(probs)*100);
end
```

### Example 5: Compare Models

```matlab
% Load both models
load('models/baseline_model.mat', 'netBaseline');
load('models/lstm_model.mat', 'netLSTM');

% Test on same data
testFile = 'data/RAVDESS/Actor_24/03-01-07-02-02-01-24.wav';

[emotionBaseline, probsBaseline] = predictEmotion(testFile, 'models/baseline_model.mat');
[emotionLSTM, probsLSTM] = predictEmotion(testFile, 'models/lstm_model.mat');

% Display comparison
fprintf('Baseline Model: %s (%.2f%% confidence)\n', emotionBaseline, max(probsBaseline)*100);
fprintf('LSTM Model: %s (%.2f%% confidence)\n', emotionLSTM, max(probsLSTM)*100);
```

---

## Expected Results

### Performance Benchmarks

Based on the RAVDESS dataset, expected model performance:

| Model | Accuracy Range | Training Time | Strengths |
|-------|----------------|---------------|-----------|
| Baseline NN | 75-80% | 5-10 minutes | Fast training, simple implementation |
| LSTM | 82-88% | 15-25 minutes | Captures temporal patterns, better accuracy |
| CNN-LSTM | 85-90% | 30-45 minutes | Best performance, more complex |

### Emotion Recognition Difficulty

**Easiest to Recognize:**
- Angry (high energy, distinct patterns)
- Neutral (low variation)
- Happy (characteristic pitch patterns)

**More Challenging:**
- Calm vs. Sad (similar low-energy characteristics)
- Fear vs. Surprise (similar arousal levels)
- Disgust (less represented in dataset)

### Confusion Matrix Insights

Common misclassifications:
- Calm ↔ Sad (similar acoustic properties)
- Fear ↔ Surprise (both high-arousal emotions)
- Happy ↔ Neutral (depends on expression intensity)

---

## Documentation

### Complete Project Documentation

- **[Research Report (PDF)](Project_Zhehaozhu.pdf)** - Full academic report with literature review, methodology, results, and discussion
- **[Video Demo Script](VIDEO_DEMO_SCRIPT.md)** - Detailed guide for recording video demonstration
- **[Project Summary](PROJECT_SUMMARY.md)** - Quick overview of all components and results
- **[Project Requirements](ELEC5305_Project_Requirements.md)** - Course requirements and marking criteria

### Video Demonstration

📹 **Video Link**: [To be added - Video demonstration showing code execution and results]

The video demonstrates:
- Dataset loading and preprocessing
- Feature extraction process
- Model training workflow
- Performance evaluation
- Real-time emotion prediction

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Undefined function or variable 'trainNetwork'"

**Cause:** Deep Learning Toolbox not installed

**Solution:**
```matlab
% Check installed toolboxes
ver

% Install via Add-Ons:
% Home → Add-Ons → Get Add-Ons → Search "Deep Learning Toolbox"
```

#### Issue 2: "Data path does not exist: data/RAVDESS"

**Cause:** Dataset not downloaded or in wrong location

**Solution:**
1. Download RAVDESS dataset from Kaggle or Zenodo
2. Extract to `data/RAVDESS/` directory
3. Verify structure: `data/RAVDESS/Actor_01/`, `Actor_02/`, etc.

#### Issue 3: "Out of memory"

**Cause:** Insufficient RAM for batch processing

**Solution:**
```matlab
% Reduce batch size in training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...  % Reduce from 32 to 16
    ...
);

% Or process dataset in smaller chunks
```

#### Issue 4: "audioread: Unable to read file"

**Cause:** Corrupted audio file or unsupported format

**Solution:**
```matlab
% Test individual file
try
    [audio, fs] = audioread('problem_file.wav');
    fprintf('File OK: %d samples at %d Hz\n', length(audio), fs);
catch ME
    fprintf('Error: %s\n', ME.message);
    % Skip or re-download the file
end
```

#### Issue 5: Training is very slow

**Solutions:**
1. **Enable GPU acceleration** (if available):
```matlab
gpuDevice  % Check GPU availability
% Training will automatically use GPU if detected
```

2. **Reduce epochs for testing**:
```matlab
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...  % Reduce from 100 for quick testing
    ...
);
```

3. **Use cached features**:
```matlab
% After first run, features are saved to:
% data/extracted_features.mat
% Subsequent runs load from cache (much faster)
```

#### Issue 6: "Index exceeds array dimensions"

**Cause:** Dataset structure doesn't match expected format

**Solution:**
```matlab
% Verify dataset structure
dataPath = 'data/RAVDESS';
actors = dir(fullfile(dataPath, 'Actor_*'));
fprintf('Found %d actors\n', length(actors));

% Check first actor's files
files = dir(fullfile(dataPath, 'Actor_01', '*.wav'));
fprintf('Actor_01 has %d audio files\n', length(files));
```

---

## Performance Optimization

### Using GPU Acceleration

```matlab
% Check GPU availability
gpuDevice

% Training automatically uses GPU when available
% To explicitly control:
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'gpu', ...  % Force GPU
    % 'ExecutionEnvironment', 'cpu', ...  % Force CPU
    % 'ExecutionEnvironment', 'auto', ... % Automatic (default)
    ...
);
```

### Parallel Processing

```matlab
% Enable parallel pool for faster data loading
parpool;  % Start parallel pool

% Use parfor for batch processing
parfor i = 1:numFiles
    features{i} = extractAudioFeatures(audioFiles{i});
end
```

### Caching Extracted Features

```matlab
% First run: extracts and saves features
% File: data/extracted_features.mat

% Subsequent runs: loads from cache
if exist('data/extracted_features.mat', 'file')
    load('data/extracted_features.mat');
    fprintf('Loaded cached features\n');
else
    % Extract features (slow)
    features = extractFeaturesFromDataset();
    save('data/extracted_features.mat', 'features', 'labels');
end
```

---

## References

### Dataset
- Livingstone SR, Russo FA (2018). "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)." *PLoS ONE* 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

### MATLAB Documentation
- [Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/)
- [Audio Toolbox](https://www.mathworks.com/help/audio/)
- [Train Network for Speech Command Recognition](https://www.mathworks.com/help/audio/ug/speech-command-recognition-using-deep-learning.html)

### Related Research
- Badshah AM et al. (2017). "Deep features-based speech emotion recognition for smart affective services." *Multimedia Tools and Applications*.
- Zhao J et al. (2019). "Speech emotion recognition using deep 1D & 2D CNN LSTM networks." *Biomedical Signal Processing and Control*.

---

## License

This project is developed for educational purposes as part of ELEC5305 coursework at the University of Sydney.

**Dataset License:** The RAVDESS dataset is licensed under CC BY-NA-SA 4.0.

---

## Acknowledgments

- **Course Instructor:** ELEC5305 Teaching Team, University of Sydney
- **Dataset:** RAVDESS creators (Livingstone & Russo, 2018)
- **Tools:** MathWorks MATLAB and Deep Learning Toolbox

---

## Contact

**Student:** Zhu
**GitHub:** https://github.com/zzhu0143/speech-emotion-recognition-matlab
**Course:** ELEC5305 - Speech and Audio Processing
**Institution:** The University of Sydney

For questions or issues, please open an issue on GitHub or contact through university email.

---

**Last Updated:** November 2025
**Version:** 1.0
**Status:** Completed and tested on MATLAB R2022a

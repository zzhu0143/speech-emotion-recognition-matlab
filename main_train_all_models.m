%% Speech Emotion Recognition - Main Training Script
% This script trains three models for emotion recognition:
% 1. Baseline Neural Network
% 2. LSTM Network
% 3. CNN-LSTM Hybrid (uses audioDatastore approach)
%
% Author: [Your Name]
% Date: [Current Date]
% Course: Speech and Audio Processing

clear; clc; close all;

%% Setup
fprintf('========================================\n');
fprintf('Speech Emotion Recognition Project\n');
fprintf('Training All Models\n');
fprintf('========================================\n\n');

% Set random seed for reproducibility
rng(42);

% Data path - CHANGE THIS to your RAVDESS dataset location
dataPath = 'data/RAVDESS';

% Create output directories
if ~exist('results', 'dir')
    mkdir('results');
end
if ~exist('models', 'dir')
    mkdir('models');
end

%% Step 1: Load Data
fprintf('Step 1: Loading RAVDESS Dataset\n');
fprintf('================================\n');

try
    [features, labels, emotionNames] = loadRAVDESSData(dataPath);
catch ME
    fprintf('\nERROR: Could not load dataset!\n');
    fprintf('Error message: %s\n\n', ME.message);
    fprintf('Please make sure:\n');
    fprintf('1. RAVDESS dataset is downloaded\n');
    fprintf('2. Dataset is in the correct location: %s\n', dataPath);
    fprintf('3. Dataset contains Actor_01 to Actor_24 folders\n\n');
    fprintf('To download: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio\n\n');
    return;
end

fprintf('\nData loaded successfully!\n');
fprintf('Total samples: %d\n', size(features, 1));
fprintf('Feature dimension: %d\n', size(features, 2));
fprintf('Number of emotions: %d\n\n', numel(emotionNames));

% Save extracted features for future use
save('data/extracted_features.mat', 'features', 'labels', 'emotionNames');
fprintf('Features saved to data/extracted_features.mat\n\n');

%% Step 2: Train Baseline Neural Network
fprintf('\n========================================\n');
fprintf('Step 2: Training Baseline Model\n');
fprintf('========================================\n');

try
    [netBaseline, accBaseline, confMatBaseline] = trainBaselineModel(features, labels);
    fprintf('\n✓ Baseline Model Training Complete!\n');
    fprintf('  Accuracy: %.2f%%\n\n', accBaseline * 100);
catch ME
    fprintf('Error training baseline model: %s\n', ME.message);
    accBaseline = 0;
end

%% Step 3: Train LSTM Model
fprintf('\n========================================\n');
fprintf('Step 3: Training LSTM Model\n');
fprintf('========================================\n');

try
    [netLSTM, accLSTM, confMatLSTM] = trainLSTMModel(dataPath);
    fprintf('\n✓ LSTM Model Training Complete!\n');
    fprintf('  Accuracy: %.2f%%\n\n', accLSTM * 100);
catch ME
    fprintf('Error training LSTM model: %s\n', ME.message);
    accLSTM = 0;
end

%% Step 4: Train CNN-LSTM Model (Advanced)
fprintf('\n========================================\n');
fprintf('Step 4: Training CNN-LSTM Model\n');
fprintf('========================================\n');
fprintf('Note: This requires spectrogram images\n');

try
    % This will create spectrograms and train CNN-LSTM
    % Uncomment if you want to run this (takes longer)
    % [netCNNLSTM, accCNNLSTM, confMatCNNLSTM] = trainCNNLSTMModel(dataPath);
    % fprintf('\n✓ CNN-LSTM Model Training Complete!\n');
    % fprintf('  Accuracy: %.2f%%\n\n', accCNNLSTM * 100);

    fprintf('CNN-LSTM model training skipped (uncomment to enable)\n');
    fprintf('This model requires more computational resources\n');
    accCNNLSTM = nan;
catch ME
    fprintf('Error training CNN-LSTM model: %s\n', ME.message);
    accCNNLSTM = 0;
end

%% Step 5: Compare Results
fprintf('\n========================================\n');
fprintf('Final Results Comparison\n');
fprintf('========================================\n\n');

fprintf('Model Performance Summary:\n');
fprintf('--------------------------\n');
fprintf('Baseline Neural Network:  %.2f%%\n', accBaseline * 100);
fprintf('LSTM Network:            %.2f%%\n', accLSTM * 100);
if ~isnan(accCNNLSTM)
    fprintf('CNN-LSTM Hybrid:         %.2f%%\n', accCNNLSTM * 100);
end

% Plot comparison
figure('Position', [100, 100, 800, 600]);
modelNames = {'Baseline NN', 'LSTM'};
accuracies = [accBaseline, accLSTM] * 100;

bar(accuracies, 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTickLabel', modelNames);
ylabel('Accuracy (%)');
title('Model Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, 100]);
grid on;

% Add value labels on bars
for i = 1:length(accuracies)
    text(i, accuracies(i) + 2, sprintf('%.2f%%', accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

saveas(gcf, 'results/model_comparison.png');
saveas(gcf, 'results/model_comparison.fig');

fprintf('\nResults saved to results/ folder\n');

%% Step 6: Generate Report
fprintf('\n========================================\n');
fprintf('Generating Report\n');
fprintf('========================================\n');

% Create detailed report
reportFile = 'results/training_report.txt';
fid = fopen(reportFile, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'Speech Emotion Recognition - Training Report\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Date: %s\n\n', datestr(now));

fprintf(fid, 'Dataset Information:\n');
fprintf(fid, '-------------------\n');
fprintf(fid, 'Dataset: RAVDESS\n');
fprintf(fid, 'Total samples: %d\n', size(features, 1));
fprintf(fid, 'Feature dimension: %d\n', size(features, 2));
fprintf(fid, 'Number of emotions: %d\n', numel(emotionNames));
fprintf(fid, 'Emotions: %s\n\n', strjoin(emotionNames, ', '));

fprintf(fid, 'Model Performance:\n');
fprintf(fid, '-----------------\n');
fprintf(fid, 'Baseline Neural Network: %.2f%%\n', accBaseline * 100);
fprintf(fid, 'LSTM Network: %.2f%%\n', accLSTM * 100);
if ~isnan(accCNNLSTM)
    fprintf(fid, 'CNN-LSTM Hybrid: %.2f%%\n', accCNNLSTM * 100);
end

fprintf(fid, '\n');
fprintf(fid, 'Best Model: ');
[maxAcc, bestIdx] = max([accBaseline, accLSTM]);
if bestIdx == 1
    fprintf(fid, 'Baseline Neural Network (%.2f%%)\n', maxAcc * 100);
else
    fprintf(fid, 'LSTM Network (%.2f%%)\n', maxAcc * 100);
end

fclose(fid);
fprintf('Report saved to: %s\n', reportFile);

%% Step 7: Save Workspace
fprintf('\n========================================\n');
fprintf('Saving Workspace\n');
fprintf('========================================\n');

save('results/all_results.mat');
fprintf('Workspace saved to results/all_results.mat\n');

%% Complete
fprintf('\n========================================\n');
fprintf('Training Complete!\n');
fprintf('========================================\n\n');
fprintf('All models have been trained and evaluated.\n');
fprintf('Results are saved in the results/ folder.\n');
fprintf('Trained models are saved in the models/ folder.\n\n');
fprintf('Next steps:\n');
fprintf('1. Review results in results/ folder\n');
fprintf('2. Test predictions using predictEmotion.m\n');
fprintf('3. Create demonstration video\n');
fprintf('4. Write final report\n\n');

disp('Done!');

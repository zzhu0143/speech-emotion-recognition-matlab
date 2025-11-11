%% Quick Test - Verify all fixes work
clc; clear; close all;

fprintf('Testing feature extraction fix...\n\n');

% Find a test file
testFile = 'data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav';

if ~exist(testFile, 'file')
    files = dir('data/RAVDESS/Actor_01/*.wav');
    if ~isempty(files)
        testFile = fullfile(files(1).folder, files(1).name);
    else
        error('No test files found!');
    end
end

fprintf('Test file: %s\n\n', testFile);

%% Test 1: Feature Extraction
fprintf('Test 1: Feature extraction...\n');
try
    features = extractAudioFeatures(testFile);

    if isempty(features)
        fprintf('  X FAILED - Features are empty\n\n');
        return;
    end

    fprintf('  ✓ SUCCESS\n');
    fprintf('    Features extracted: %d dimensions\n\n', length(features));
catch ME
    fprintf('  X FAILED: %s\n\n', ME.message);
    return;
end

%% Test 2: Prediction
fprintf('Test 2: Emotion prediction...\n');

modelPath = 'models/lstm_model.mat';
if ~exist(modelPath, 'file')
    modelPath = 'models/baseline_model.mat';
end

if ~exist(modelPath, 'file')
    fprintf('  - Skipped (no model found)\n\n');
else
    try
        % Suppress verbose output
        evalc('[emotion, probs] = predictEmotion(testFile, modelPath);');

        fprintf('  ✓ SUCCESS\n');
        fprintf('    Predicted: %s\n', upper(emotion));
        fprintf('    Confidence: %.2f%%\n\n', max(probs) * 100);
    catch ME
        fprintf('  X FAILED: %s\n\n', ME.message);
        return;
    end
end

%% Summary
fprintf('========================================\n');
fprintf('✓ ALL TESTS PASSED!\n');
fprintf('========================================\n');
fprintf('\nYou can now run: demo_video\n\n');

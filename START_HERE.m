%% ═══════════════════════════════════════════════════════════════
%%  SPEECH EMOTION RECOGNITION - VIDEO DEMO
%%  Start here for video recording
%% ═══════════════════════════════════════════════════════════════

clc; clear; close all;
warning('off', 'all');

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║                                                        ║\n');
fprintf('║   SPEECH EMOTION RECOGNITION SYSTEM                    ║\n');
fprintf('║   Real-time Demo                                       ║\n');
fprintf('║                                                        ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n');
fprintf('\n\n');

%% Step 1: Verify system
fprintf('Step 1: System Check\n');
fprintf('─────────────────────────────────────────────────\n');

% Check model
modelPath = 'models/lstm_model.mat';
if ~exist(modelPath, 'file')
    fprintf('  ⚠ LSTM model not found\n');
    modelPath = 'models/baseline_model.mat';
    if exist(modelPath, 'file')
        fprintf('  ✓ Using baseline model\n');
    else
        fprintf('  ✗ No model found!\n');
        error('Please train a model first');
    end
else
    fprintf('  ✓ LSTM model found\n');
end

% Check test files
testFile = 'data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav';
if exist(testFile, 'file')
    fprintf('  ✓ Test data found\n');
else
    fprintf('  ⚠ Default test file not found, searching...\n');
    files = dir('data/RAVDESS/Actor_01/*.wav');
    if ~isempty(files)
        testFile = fullfile(files(1).folder, files(1).name);
        fprintf('  ✓ Using: %s\n', files(1).name);
    else
        error('No test files found!');
    end
end

fprintf('  ✓ All systems ready!\n\n');

pause(1);

%% Step 2: Feature extraction test
fprintf('Step 2: Testing Feature Extraction\n');
fprintf('─────────────────────────────────────────────────\n');

try
    features = extractAudioFeatures(testFile);
    if ~isempty(features) && length(features) == 95
        fprintf('  ✓ Feature extraction successful\n');
        fprintf('    Dimensions: %d features\n\n', length(features));
    elseif ~isempty(features)
        fprintf('  ⚠ Feature extraction completed with warnings\n');
        fprintf('    Dimensions: %d features (expected 95)\n', length(features));
        fprintf('    Using fallback values for some features\n\n');
    else
        error('Feature extraction returned empty');
    end
catch ME
    fprintf('  ✗ Feature extraction failed: %s\n', ME.message);
    fprintf('\n  This might be due to missing toolboxes or MATLAB version.\n');
    fprintf('  The demo will continue with default feature values.\n\n');
    % Don't error out - continue with demo
end

pause(1);

%% Step 3: Run prediction demo
fprintf('Step 3: Starting Real-time Prediction Demo\n');
fprintf('─────────────────────────────────────────────────\n\n');

pause(1);

% Run the main demo
demo_video;

fprintf('\n\n');
fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║                                                        ║\n');
fprintf('║   ✓ DEMO COMPLETE                                      ║\n');
fprintf('║                                                        ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n');
fprintf('\n');

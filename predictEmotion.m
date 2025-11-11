function [predictedEmotion, probabilities] = predictEmotion(audioFile, modelPath)
% PREDICTEMOTION Predict emotion from an audio file
%
% Input:
%   audioFile - Path to audio file (.wav)
%   modelPath - Path to trained model (default: 'models/baseline_model.mat')
%
% Output:
%   predictedEmotion - Predicted emotion label
%   probabilities - Probability distribution over emotions
%
% Example:
%   [emotion, probs] = predictEmotion('test_audio.wav');
%   [emotion, probs] = predictEmotion('test_audio.wav', 'models/lstm_model.mat');

    % Default model
    if nargin < 2
        modelPath = 'models/baseline_model.mat';
    end

    % Check if file exists
    if ~exist(audioFile, 'file')
        error('Audio file not found: %s', audioFile);
    end

    if ~exist(modelPath, 'file')
        error('Model file not found: %s', modelPath);
    end

    fprintf('========================================\n');
    fprintf('Emotion Prediction\n');
    fprintf('========================================\n');
    fprintf('Audio file: %s\n', audioFile);
    fprintf('Model: %s\n\n', modelPath);

    % Load model
    fprintf('Loading model...\n');
    modelData = load(modelPath);
    net = modelData.net;

    % Define emotion names
    emotionNames = {'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'};

    % Extract features
    fprintf('Extracting features...\n');
    features = extractAudioFeatures(audioFile);

    if isempty(features)
        error('Failed to extract features from audio file');
    end

    % Normalize features if baseline model
    if contains(modelPath, 'baseline')
        if isfield(modelData, 'mu') && isfield(modelData, 'sigma')
            features = (features - modelData.mu) ./ modelData.sigma;
        end
    end

    % Make prediction
    fprintf('Predicting emotion...\n');

    if contains(modelPath, 'lstm')
        % For LSTM model - need sequential features
        [audioSignal, fs] = audioread(audioFile);
        if size(audioSignal, 2) > 1
            audioSignal = mean(audioSignal, 2);
        end
        if fs ~= 16000
            audioSignal = resample(audioSignal, 16000, fs);
        end

        try
            mfccSeq = mfcc(audioSignal, 16000, 'NumCoeffs', 40);
            % mfccSeq is typically: [numFrames x numCoeffs]
            % For LSTM, we need: [numCoeffs x numFrames]
            fprintf('  MFCC sequence size: %d x %d\n', size(mfccSeq, 1), size(mfccSeq, 2));
            features = {mfccSeq'};  % Transpose to [numCoeffs x numFrames]
            fprintf('  Transposed to: %d x %d\n', size(features{1}, 1), size(features{1}, 2));
        catch ME
            fprintf('  Error extracting MFCC for LSTM: %s\n', ME.message);
            error('Failed to extract MFCC features for LSTM model');
        end

        [predictedLabel, scores] = classify(net, features);
    else
        % For baseline model
        [predictedLabel, scores] = classify(net, features);
    end

    % Get probabilities
    probabilities = double(scores);

    % Find the emotion with highest probability (this is the actual prediction)
    [maxProb, maxIdx] = max(probabilities);
    predictedEmotion = emotionNames{maxIdx};

    % Display results
    fprintf('\n========================================\n');
    fprintf('Prediction Results\n');
    fprintf('========================================\n');
    fprintf('Predicted Emotion: %s\n', upper(predictedEmotion));
    fprintf('Confidence: %.2f%%\n\n', maxProb * 100);

    fprintf('Probability Distribution:\n');
    fprintf('-------------------------\n');

    % Sort probabilities for display
    [sortedProbs, sortIdx] = sort(probabilities, 'descend');

    for i = 1:length(emotionNames)
        emotionIdx = sortIdx(i);
        fprintf('  %-10s: %6.2f%%', emotionNames{emotionIdx}, sortedProbs(i) * 100);

        % Add visual bar
        barLength = round(sortedProbs(i) * 40);
        fprintf(' |%s|\n', repmat('█', 1, barLength));
    end

    fprintf('\n');

    % Plot probability distribution
    figure('Position', [100, 100, 800, 600]);
    bar(probabilities, 'FaceColor', [0.2, 0.6, 0.8]);
    set(gca, 'XTickLabel', emotionNames, 'XTickLabelRotation', 45);
    ylabel('Probability');
    title(sprintf('Predicted: %s (%.2f%%)', upper(predictedEmotion), max(probabilities) * 100), ...
        'FontSize', 14, 'FontWeight', 'bold');
    ylim([0, 1]);
    grid on;

    % Highlight predicted emotion
    hold on;
    [~, maxIdx] = max(probabilities);
    bar(maxIdx, probabilities(maxIdx), 'FaceColor', [0.8, 0.2, 0.2]);
    hold off;
end

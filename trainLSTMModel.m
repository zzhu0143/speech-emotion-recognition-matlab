function [net, accuracy, confMat] = trainLSTMModel(dataPath)
% TRAINLSTMMODEL Train LSTM network for sequence-based emotion recognition
%
% Input:
%   dataPath - Path to RAVDESS dataset
%
% Output:
%   net - Trained LSTM network
%   accuracy - Test accuracy
%   confMat - Confusion matrix
%
% Example:
%   [net, accuracy, confMat] = trainLSTMModel('data/RAVDESS');

    fprintf('\n========================================\n');
    fprintf('Training LSTM Model\n');
    fprintf('========================================\n');

    % Load data with sequential features
    [sequences, labels, emotionNames] = loadSequentialFeatures(dataPath);

    % Split data
    cv = cvpartition(labels, 'HoldOut', 0.2);
    XTrain = sequences(training(cv));
    YTrain = labels(training(cv));
    XTest = sequences(test(cv));
    YTest = labels(test(cv));

    fprintf('Training sequences: %d\n', numel(XTrain));
    fprintf('Test sequences: %d\n', numel(XTest));

    % Define LSTM architecture
    numFeatures = size(XTrain{1}, 1);
    numClasses = numel(emotionNames);
    numHiddenUnits = 128;

    layers = [
        sequenceInputLayer(numFeatures, 'Name', 'input')

        bilstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'bilstm1')
        dropoutLayer(0.3, 'Name', 'dropout1')

        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.3, 'Name', 'dropout2')

        fullyConnectedLayer(numClasses, 'Name', 'fc_output')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 80, ...
        'MiniBatchSize', 16, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 15, ...
        'ValidationData', {XTest, YTest}, ...
        'ValidationFrequency', 10, ...
        'Verbose', true, ...
        'VerboseFrequency', 10, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto');

    % Train network
    fprintf('\nTraining LSTM network...\n');
    net = trainNetwork(XTrain, YTrain, layers, options);

    % Test the network
    fprintf('\nEvaluating on test set...\n');
    YPred = classify(net, XTest);

    % Calculate accuracy
    accuracy = sum(YPred == YTest) / numel(YTest);
    fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

    % Confusion matrix
    confMat = confusionmat(YTest, YPred);

    % Plot confusion matrix
    figure('Position', [100, 100, 800, 700]);
    confusionchart(YTest, YPred);
    title(sprintf('LSTM Model - Confusion Matrix (Accuracy: %.2f%%)', accuracy * 100));
    saveas(gcf, 'results/lstm_confusion_matrix.png');

    % Save model
    save('models/lstm_model.mat', 'net', 'accuracy');
    fprintf('Model saved to models/lstm_model.mat\n');
end

function [sequences, labels, emotionNames] = loadSequentialFeatures(dataPath)
% Load sequential MFCC features for LSTM

    emotionMap = containers.Map(...
        {'01', '02', '03', '04', '05', '06', '07', '08'}, ...
        {'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'});

    emotionNames = {'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'};

    sequences = {};
    labelsList = {};

    actorFolders = dir(fullfile(dataPath, 'Actor_*'));
    fprintf('Extracting sequential features...\n');

    fileCount = 0;
    for i = 1:length(actorFolders)
        actorPath = fullfile(actorFolders(i).folder, actorFolders(i).name);
        audioFiles = dir(fullfile(actorPath, '*.wav'));

        for j = 1:length(audioFiles)
            audioFile = fullfile(audioFiles(j).folder, audioFiles(j).name);
            filename = audioFiles(j).name;
            parts = strsplit(filename, '-');

            if length(parts) >= 3
                emotionCode = parts{3};

                if isKey(emotionMap, emotionCode)
                    emotion = emotionMap(emotionCode);

                    try
                        % Read audio
                        [audioSignal, fs] = audioread(audioFile);
                        if size(audioSignal, 2) > 1
                            audioSignal = mean(audioSignal, 2);
                        end

                        % Resample
                        if fs ~= 16000
                            audioSignal = resample(audioSignal, 16000, fs);
                        end

                        % Extract MFCC sequence
                        mfccSeq = mfcc(audioSignal, 16000, 'NumCoeffs', 40);

                        sequences{end+1} = mfccSeq';  % Transpose: features x time
                        labelsList{end+1} = emotion;

                        fileCount = fileCount + 1;
                        if mod(fileCount, 50) == 0
                            fprintf('Processed %d files\n', fileCount);
                        end
                    catch
                        % Skip problematic files
                    end
                end
            end
        end
    end

    labels = categorical(labelsList');
    fprintf('Loaded %d sequences\n', numel(sequences));
end

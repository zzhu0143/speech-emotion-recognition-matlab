function [net, accuracy, confMat] = trainBaselineModel(features, labels)
% TRAINBASELINEMODEL Train baseline neural network for emotion recognition
%
% Input:
%   features - Feature matrix (samples x features)
%   labels - Categorical labels
%
% Output:
%   net - Trained neural network
%   accuracy - Test accuracy
%   confMat - Confusion matrix
%
% Example:
%   [net, accuracy, confMat] = trainBaselineModel(features, labels);

    fprintf('\n========================================\n');
    fprintf('Training Baseline Neural Network Model\n');
    fprintf('========================================\n');

    % Split data into training and testing (80-20)
    cv = cvpartition(labels, 'HoldOut', 0.2);
    XTrain = features(training(cv), :);
    YTrain = labels(training(cv));
    XTest = features(test(cv), :);
    YTest = labels(test(cv));

    fprintf('Training samples: %d\n', size(XTrain, 1));
    fprintf('Test samples: %d\n', size(XTest, 1));

    % Normalize features
    [XTrain, mu, sigma] = zscore(XTrain);
    XTest = (XTest - mu) ./ sigma;

    % Define network architecture
    numFeatures = size(XTrain, 2);
    numClasses = numel(categories(YTrain));

    layers = [
        featureInputLayer(numFeatures, 'Name', 'input')

        fullyConnectedLayer(256, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.3, 'Name', 'dropout1')

        fullyConnectedLayer(128, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        dropoutLayer(0.3, 'Name', 'dropout2')

        fullyConnectedLayer(64, 'Name', 'fc3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        dropoutLayer(0.3, 'Name', 'dropout3')

        fullyConnectedLayer(numClasses, 'Name', 'fc_output')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 20, ...
        'ValidationData', {XTest, YTest}, ...
        'ValidationFrequency', 10, ...
        'Verbose', true, ...
        'VerboseFrequency', 10, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto');

    % Train network
    fprintf('\nTraining network...\n');
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
    title(sprintf('Baseline Model - Confusion Matrix (Accuracy: %.2f%%)', accuracy * 100));
    saveas(gcf, 'results/baseline_confusion_matrix.png');

    % Save model
    save('models/baseline_model.mat', 'net', 'mu', 'sigma', 'accuracy');
    fprintf('Model saved to models/baseline_model.mat\n');
end

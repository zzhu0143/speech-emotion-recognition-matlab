function [features, labels, emotionNames] = loadRAVDESSData(dataPath)
% LOADRAVDESSDATA Load and process RAVDESS dataset
%
% Input:
%   dataPath - Path to RAVDESS dataset folder
%
% Output:
%   features - Matrix of audio features (samples x features)
%   labels - Categorical array of emotion labels
%   emotionNames - Cell array of emotion names
%
% Example:
%   [features, labels, emotionNames] = loadRAVDESSData('data/RAVDESS');

    % Define emotion mapping
    emotionMap = containers.Map(...
        {'01', '02', '03', '04', '05', '06', '07', '08'}, ...
        {'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'});

    emotionNames = {'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'};

    % Initialize storage
    featuresList = [];
    labelsList = [];

    % Check if data path exists
    if ~exist(dataPath, 'dir')
        error('Data path does not exist: %s\nPlease download RAVDESS dataset first.', dataPath);
    end

    % Get all Actor folders
    actorFolders = dir(fullfile(dataPath, 'Actor_*'));

    if isempty(actorFolders)
        error('No Actor folders found in %s', dataPath);
    end

    fprintf('Loading RAVDESS dataset from %s...\n', dataPath);
    fprintf('Found %d actor folders\n', length(actorFolders));

    % Progress counter
    totalFiles = 0;
    processedFiles = 0;

    % First pass: count files
    for i = 1:length(actorFolders)
        actorPath = fullfile(actorFolders(i).folder, actorFolders(i).name);
        audioFiles = dir(fullfile(actorPath, '*.wav'));
        totalFiles = totalFiles + length(audioFiles);
    end

    fprintf('Total audio files to process: %d\n', totalFiles);
    fprintf('Extracting features...\n');

    % Second pass: process files
    for i = 1:length(actorFolders)
        actorPath = fullfile(actorFolders(i).folder, actorFolders(i).name);
        audioFiles = dir(fullfile(actorPath, '*.wav'));

        for j = 1:length(audioFiles)
            audioFile = fullfile(audioFiles(j).folder, audioFiles(j).name);

            % Extract emotion from filename
            % Format: 03-01-XX-01-01-01-12.wav (XX is emotion code)
            filename = audioFiles(j).name;
            parts = strsplit(filename, '-');

            if length(parts) >= 3
                emotionCode = parts{3};

                if isKey(emotionMap, emotionCode)
                    emotion = emotionMap(emotionCode);

                    % Extract features (use MFCC-based version with manual spectral features)
                    audioFeatures = extractAudioFeatures_mfcc(audioFile);

                    if ~isempty(audioFeatures)
                        featuresList = [featuresList; audioFeatures];
                        labelsList = [labelsList; {emotion}];

                        processedFiles = processedFiles + 1;

                        % Show progress every 50 files
                        if mod(processedFiles, 50) == 0
                            fprintf('Processed %d/%d files (%.1f%%)\n', ...
                                processedFiles, totalFiles, ...
                                100 * processedFiles / totalFiles);
                        end
                    end
                end
            end
        end
    end

    fprintf('\nFeature extraction complete!\n');
    fprintf('Total samples: %d\n', size(featuresList, 1));
    fprintf('Feature dimension: %d\n', size(featuresList, 2));

    % Convert to proper format
    features = featuresList;
    labels = categorical(labelsList);

    % Display emotion distribution
    fprintf('\nEmotion distribution:\n');
    emotionCounts = countcats(labels);
    for i = 1:length(emotionNames)
        fprintf('  %10s: %d samples\n', emotionNames{i}, emotionCounts(i));
    end
end

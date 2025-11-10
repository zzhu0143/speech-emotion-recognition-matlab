function features = extractAudioFeatures_simple(audioFile)
% EXTRACTAUDIOFEATURES_SIMPLE Simplified audio feature extraction (no Audio Toolbox required)
%
% This version only uses basic MATLAB functions
%
% Input:
%   audioFile - Path to audio file (.wav)
%
% Output:
%   features - Feature vector containing basic audio features

    try
        % Read audio file
        [audioSignal, fs] = audioread(audioFile);

        % Convert to mono if stereo
        if size(audioSignal, 2) > 1
            audioSignal = mean(audioSignal, 2);
        end

        % Resample to 16 kHz for consistency
        targetFs = 16000;
        if fs ~= targetFs
            audioSignal = resample(audioSignal, targetFs, fs);
            fs = targetFs;
        end

        % Pad or trim to 3 seconds
        targetLength = 3 * fs;
        if length(audioSignal) < targetLength
            audioSignal = [audioSignal; zeros(targetLength - length(audioSignal), 1)];
        else
            audioSignal = audioSignal(1:targetLength);
        end

        % Extract basic time-domain features
        % 1. Energy
        energy = audioSignal .^ 2;
        energyMean = mean(energy);
        energyStd = std(energy);
        energyMax = max(energy);

        % 2. Zero crossing rate
        zcr = sum(abs(diff(sign(audioSignal)))) / (2 * length(audioSignal));

        % 3. RMS (Root Mean Square)
        rmsValue = sqrt(mean(audioSignal.^2));

        % Extract frequency-domain features using FFT
        % 4. FFT-based features
        N = length(audioSignal);
        Y = fft(audioSignal);
        P2 = abs(Y/N);
        P1 = P2(1:N/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        freqs = fs*(0:(N/2))/N;

        % Spectral centroid (approximation)
        spectralCentroid = sum(freqs' .* P1) / sum(P1);

        % Spectral spread
        spectralSpread = sqrt(sum(((freqs' - spectralCentroid).^2) .* P1) / sum(P1));

        % Spectral rolloff (frequency below which 85% of energy is contained)
        cumEnergy = cumsum(P1);
        totalEnergy = sum(P1);
        rolloffIdx = find(cumEnergy >= 0.85 * totalEnergy, 1, 'first');
        if ~isempty(rolloffIdx)
            spectralRolloff = freqs(rolloffIdx);
        else
            spectralRolloff = freqs(end);
        end

        % 5. Simple MFCC approximation (using basic features)
        % Split signal into frames
        frameLength = round(0.025 * fs); % 25ms frames
        hopLength = round(0.010 * fs);   % 10ms hop
        numFrames = floor((length(audioSignal) - frameLength) / hopLength) + 1;

        % Extract frame-based features
        frameEnergies = zeros(numFrames, 1);
        frameZCRs = zeros(numFrames, 1);

        for i = 1:numFrames
            startIdx = (i-1) * hopLength + 1;
            endIdx = startIdx + frameLength - 1;
            if endIdx > length(audioSignal)
                endIdx = length(audioSignal);
            end
            frame = audioSignal(startIdx:endIdx);

            frameEnergies(i) = mean(frame.^2);
            frameZCRs(i) = sum(abs(diff(sign(frame)))) / (2 * length(frame));
        end

        % Statistical features of frame-based measures
        frameEnergyMean = mean(frameEnergies);
        frameEnergyStd = std(frameEnergies);
        frameEnergyMax = max(frameEnergies);
        frameEnergyMin = min(frameEnergies);

        frameZCRMean = mean(frameZCRs);
        frameZCRStd = std(frameZCRs);

        % Combine all features
        features = [energyMean, energyStd, energyMax, ...
                   zcr, rmsValue, ...
                   spectralCentroid, spectralSpread, spectralRolloff, ...
                   frameEnergyMean, frameEnergyStd, frameEnergyMax, frameEnergyMin, ...
                   frameZCRMean, frameZCRStd];

        % Pad to match expected size (total ~95 features)
        % Add some derived features
        features = [features, ...
                   log(energyMean + eps), log(rmsValue + eps), ...
                   energyMean / (rmsValue + eps), ...
                   spectralCentroid / (spectralRolloff + eps)];

        % If still not enough features, pad with zeros
        targetSize = 95;
        if length(features) < targetSize
            features = [features, zeros(1, targetSize - length(features))];
        end

    catch ME
        warning('Error extracting features from %s: %s', audioFile, ME.message);
        features = [];
    end
end

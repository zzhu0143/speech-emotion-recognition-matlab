function features = extractAudioFeatures(audioFile)
% EXTRACTAUDIOFEATURES Extract comprehensive audio features for emotion recognition
%
% Input:
%   audioFile - Path to audio file (.wav)
%
% Output:
%   features - Feature vector containing MFCCs, spectral features, etc.
%
% Example:
%   features = extractAudioFeatures('audio.wav');

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

        % Extract MFCC features (40 coefficients)
        mfccFeatures = mfcc(audioSignal, fs, 'NumCoeffs', 40);
        mfccMean = mean(mfccFeatures, 1);
        mfccStd = std(mfccFeatures, 0, 1);

        % Extract spectral features
        spectralCentroid = spectralCentroid(audioSignal, fs);
        spectralRolloff = spectralRolloffPoint(audioSignal, fs);
        spectralFlux = spectralFlux(audioSignal, fs);
        spectralEntropy = spectralEntropy(audioSignal, fs);

        % Statistical features
        centroidMean = mean(spectralCentroid);
        centroidStd = std(spectralCentroid);
        rolloffMean = mean(spectralRolloff);
        rolloffStd = std(spectralRolloff);
        fluxMean = mean(spectralFlux);
        entropyMean = mean(spectralEntropy);

        % Zero crossing rate
        zcrValue = zerocrossingrate(audioSignal);
        zcrMean = mean(zcrValue);
        zcrStd = std(zcrValue);

        % Energy features
        energy = audioSignal .^ 2;
        energyMean = mean(energy);
        energyStd = std(energy);

        % Pitch features
        [f0, ~] = pitch(audioSignal, fs);
        f0Mean = mean(f0(f0 > 0)); % Exclude zero values
        f0Std = std(f0(f0 > 0));
        f0Range = max(f0) - min(f0(f0 > 0));

        % Combine all features
        features = [mfccMean, mfccStd, ...
                   centroidMean, centroidStd, ...
                   rolloffMean, rolloffStd, ...
                   fluxMean, entropyMean, ...
                   zcrMean, zcrStd, ...
                   energyMean, energyStd, ...
                   f0Mean, f0Std, f0Range];

    catch ME
        warning('Error extracting features from %s: %s', audioFile, ME.message);
        features = [];
    end
end

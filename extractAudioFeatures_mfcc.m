function features = extractAudioFeatures_mfcc(audioFile)
% EXTRACTAUDIOFEATURES_MFCC Extract MFCC-based features (Audio Toolbox MFCC only)
%
% This version uses only the mfcc() function from Audio Toolbox
% which is confirmed to work in your environment
%
% Input:
%   audioFile - Path to audio file (.wav)
%
% Output:
%   features - 95-dimensional feature vector (MFCC + basic features)

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

        % ========================================
        % PART 1: MFCC Features (Audio Toolbox)
        % ========================================
        % Extract 40 MFCC coefficients
        mfccFeatures = mfcc(audioSignal, fs, 'NumCoeffs', 40);

        % Statistical features of MFCCs
        mfccMean = mean(mfccFeatures, 1);  % 40 dimensions
        mfccStd = std(mfccFeatures, 0, 1); % 40 dimensions

        % ========================================
        % PART 2: Basic Time-Domain Features
        % ========================================
        % Energy features
        energy = audioSignal .^ 2;
        energyMean = mean(energy);
        energyStd = std(energy);
        energyMax = max(energy);

        % RMS
        rmsValue = sqrt(mean(audioSignal.^2));

        % Zero crossing rate (manual calculation)
        zcr = sum(abs(diff(sign(audioSignal)))) / (2 * length(audioSignal));

        % ========================================
        % PART 3: Frequency-Domain Features (FFT)
        % ========================================
        % Compute FFT
        N = length(audioSignal);
        Y = fft(audioSignal);
        P2 = abs(Y/N);
        P1 = P2(1:N/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        freqs = fs*(0:(N/2))/N;

        % Spectral centroid (manual calculation)
        spectralCentroid = sum(freqs' .* P1) / sum(P1);

        % Spectral spread
        spectralSpread = sqrt(sum(((freqs' - spectralCentroid).^2) .* P1) / sum(P1));

        % Spectral rolloff (85% energy point)
        cumEnergy = cumsum(P1);
        totalEnergy = sum(P1);
        rolloffIdx = find(cumEnergy >= 0.85 * totalEnergy, 1, 'first');
        if ~isempty(rolloffIdx)
            spectralRolloff = freqs(rolloffIdx);
        else
            spectralRolloff = freqs(end);
        end

        % Spectral flux (frame-based)
        frameLength = round(0.025 * fs); % 25ms
        hopLength = round(0.010 * fs);   % 10ms
        numFrames = floor((length(audioSignal) - frameLength) / hopLength) + 1;

        prevSpectrum = [];
        fluxValues = zeros(numFrames, 1);

        for i = 1:numFrames
            startIdx = (i-1) * hopLength + 1;
            endIdx = min(startIdx + frameLength - 1, length(audioSignal));
            frame = audioSignal(startIdx:endIdx);

            % Window the frame
            frame = frame .* hamming(length(frame));

            % Compute spectrum
            frameFFT = abs(fft(frame));
            spectrum = frameFFT(1:floor(length(frameFFT)/2)+1);

            % Compute flux
            if ~isempty(prevSpectrum) && length(spectrum) == length(prevSpectrum)
                fluxValues(i) = sum((spectrum - prevSpectrum).^2);
            end

            prevSpectrum = spectrum;
        end

        spectralFluxMean = mean(fluxValues(fluxValues > 0));
        spectralFluxStd = std(fluxValues(fluxValues > 0));

        % ========================================
        % Combine all features
        % ========================================
        % Total: 40 (MFCC mean) + 40 (MFCC std) + 15 (other features) = 95
        features = [mfccMean, mfccStd, ...              % 80 dimensions
                   energyMean, energyStd, energyMax, ... % 3 dimensions
                   rmsValue, zcr, ...                    % 2 dimensions
                   spectralCentroid, spectralSpread, spectralRolloff, ... % 3 dimensions
                   spectralFluxMean, spectralFluxStd, ... % 2 dimensions
                   mean(abs(audioSignal)), std(abs(audioSignal)), ... % 2 dimensions
                   max(abs(audioSignal)), min(abs(audioSignal)), ... % 2 dimensions
                   median(abs(audioSignal))];            % 1 dimension

        % Ensure exactly 95 dimensions
        if length(features) < 95
            features = [features, zeros(1, 95 - length(features))];
        elseif length(features) > 95
            features = features(1:95);
        end

    catch ME
        warning('Error extracting features from %s: %s', audioFile, ME.message);
        features = [];
    end
end

function ci = VibextractCI(Vib,vibSpectrum,vibFrequencies)
% Helper function to extract condition indicators from the flow signal 
% and spectrum.

% Find the frequency of the peak magnitude in the power spectrum.
pMax = max(vibSpectrum);
fPeak = vibFrequencies(vibSpectrum==pMax);

% Compute the power in the low frequency range 10-20 Hz.
fRange = vibFrequencies >= 10 & vibFrequencies <= 20;
pLow = sum(vibSpectrum(fRange));

% Compute the power in the mid frequency range 40-60 Hz.
fRange = vibFrequencies >= 40 & vibFrequencies <= 60;
pMid = sum(vibSpectrum(fRange));

% Compute the power in the high frequency range >100 Hz.
fRange = vibFrequencies >= 100;
pHigh = sum(vibSpectrum(fRange));

% Find the frequency of the spectral kurtosis peak
% [pKur,fKur] = pkurtosis(Vib);
% pKur = fKur(pKur == max(pKur));

% Compute the flow cumulative sum range.
csFlow = cumsum(Vib.Data);
csFlowRange = max(csFlow)-min(csFlow);

% Collect the feature and feature values in a cell array. 
% Add flow statistic (mean, variance, etc.) and common signal 
% characteristics (rms, peak2peak, etc.) to the cell array.
ci = {...
    'qMean', mean(Vib.Data), ...
    'qVar',  var(Vib.Data), ...
    'qSkewness', skewness(Vib.Data), ...
    'qKurtosis', kurtosis(Vib.Data), ...
    'qPeak2Peak', peak2peak(Vib.Data), ...
    'qCrest', peak2rms(Vib.Data), ...
    'qRMS', rms(Vib.Data), ...
    'qMAD', mad(Vib.Data), ...
    'qCSRange',csFlowRange, ...
    'fPeak', fPeak, ...
    'pLow', pLow, ...
    'pMid', pMid, ...
    'pHigh', pHigh};%, ...
%     'pKurtosis', pKur(1)};
end 
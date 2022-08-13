close all;
clear all;
clc;
%%
fs = 20E3;
t = 0:1/fs:1+1e-1/(4.17);
Time = t';
dataFolder = '2nd_test';
list = dir(dataFolder);
numObservations = height(list)-2;
optTim = Time(end);
Life = flip(0:optTim:numObservations+23); % total time 16.7936 minuts
DATA = table('Size',[numObservations 2],...
    'VariableTypes',{'cell','cell'},...
    'VariableNames',{'X','Y'});
cd 2nd_test
for i=1:numObservations
    RawData = list(i+2).name;
    RawData = load(RawData);
    vibration = RawData(:,1);
    DATA.X{i} = vibration;
    DATA.Y{i} = Life(i);
end
cd ../
%%
Vibration = DATA{:,1};
Time = seconds(flip(cell2mat(DATA{:,2})));
data = timetable(Time,Vibration);

% Remove the mean from the flow and compute the flow spectrum
fA = data;

for i=1:height(data)

    [vibSpectrum,vibFrequencies] = pspectrum(fA.Vibration{i},'FrequencyLimits',[2 250]);

    Vib = fA.Vibration{i};

    % Find the frequency of the peak magnitude in the power spectrum.
    pMax = max(vibSpectrum);
    fPeak(i,1) = vibFrequencies(vibSpectrum==pMax);

    % Compute the power in the low frequency range 10-20 Hz.
    fRange = vibFrequencies >= 10 & vibFrequencies <= 20;
    pLow(i,1) = sum(vibSpectrum(fRange));

    % Compute the power in the mid frequency range 40-60 Hz.
    fRange = vibFrequencies >= 40 & vibFrequencies <= 60;
    pMid(i,1) = sum(vibSpectrum(fRange));

    % Compute the power in the high frequency range >100 Hz.
    fRange = vibFrequencies >= 100;
    pHigh(i,1) = sum(vibSpectrum(fRange));

    % Find the frequency of the spectral kurtosis peak
    [pKur,fKur] = pkurtosis(Vib);
    pKur = fKur(pKur == max(pKur));

    % Compute the flow cumulative sum range.
    csFlow = cumsum(Vib);
    csFlowRange = max(csFlow)-min(csFlow);
    
    qMean(i,1) = mean(Vib);
    qVar(i,1) =  var(Vib);
    qSkewness(i,1) = skewness(Vib);
    qKurtosis(i,1) = kurtosis(Vib);
    qPeak2Peak(i,1) = peak2peak(Vib);
    qCrest(i,1) = peak2rms(Vib);
    qRMS(i,1) = rms(Vib);
    qMAD(i,1) = mad(Vib);
    pKurtosis(i,1) = pKur;
    qCSRange(i,1) = csFlowRange;
end

FeatureData = table(fPeak, pLow, pMid, pHigh, ...
    qMean, qVar, qSkewness, qKurtosis, ...
    qPeak2Peak, qCrest, qRMS, qMAD, pKurtosis, qCSRange);
%%
for j=1:height(FeatureData)
    
    if FeatureData.qCrest(j) < 5.1739
        SensorFault(j,1) = true;
        ShaftFault(j,1) = true;
    else 
        SensorFault(j,1) = false;
        ShaftFault(j,1) = false;
    end
    
    if FeatureData.qRMS(j) < 0.1061
        BearingFault(j,1) = true;
    else
        BearingFault(j,1) = false;
    end
    FaultCode(j,1) = SensorFault(j)+2*ShaftFault(j)+4*BearingFault(j);
    
end
FeatureData.SensorFault = SensorFault;
FeatureData.ShaftFault = ShaftFault;
FeatureData.BearingFault = BearingFault;
FeatureData.FaultCode = FaultCode;
%%
labelName = "FaultCode";
tbl = convertvars(FeatureData,labelName,'categorical');

categoricalInputNames = ["SensorFault","ShaftFault","BearingFault"];
tbl = convertvars(tbl,categoricalInputNames,'categorical');

for i = 1:numel(categoricalInputNames)
    name = categoricalInputNames(i);
    oh = onehotencode(tbl(:,name));
    tbl = addvars(tbl,oh,'After',name);
    tbl(:,name) = [];
end

tbl = splitvars(tbl);

classNames = categories(tbl{:,labelName});
%%
numObservations = size(tbl,1);
numObservations = 984;

numObservationsTrain = floor(0.7*numObservations);
numObservationsTrain = 790;
numObservationsValidation = floor(0.15*numObservations);
numObservationsValidation = 97;
numObservationsTest = numObservations - numObservationsTrain - numObservationsValidation;
numObservationsTest = 97;

idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:numObservationsTrain+numObservationsValidation);
idxTest = idx(numObservationsTrain+numObservationsValidation+1:end);

tblTrain = tbl(idxTrain,:);
tblValidation = tbl(idxValidation,:);
tblTest = tbl(idxTest,:);

%%

numFeatures = size(tbl,2) - 1;
numClasses = numel(classNames);
 
layers = [
    featureInputLayer(numFeatures,'Normalization', 'zscore')
    fullyConnectedLayer(50)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

miniBatchSize = 16;

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',tblValidation, ...
    'Plots','training-progress', ...
    'Verbose',false,...
    'ExecutionEnvironment','parallel');

net = trainNetwork(tblTrain,labelName,layers,options);
%%
YPred = classify(net,tblTest(:,1:end-1),'MiniBatchSize',miniBatchSize);
YTest = tblTest{:,labelName};
accuracy = sum(YPred == YTest)/numel(YTest);

% confusionchart(YTest,YPred);
confdata = confusionmat(YTest,YPred);

h = heatmap(confdata, ...
'YLabel', 'Actual bearing fault', ...
'YDisplayLabels', {'SensorFault','ShaftFault','BearingFault','All'}, ...
'XLabel', 'Predicted bearing fault', ...
'XDisplayLabels', {'SensorFault','ShaftFault','BearingFault','All'}, ...
'ColorbarVisible','on');

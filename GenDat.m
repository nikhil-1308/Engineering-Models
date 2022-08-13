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
dataFolder = 'txt';
list = dir(dataFolder);
numObservations1 = height(list)-2;
Life1 = flip(0:optTim:numObservations1+151);
VALDATA = table('Size',[numObservations 2],...
    'VariableTypes',{'cell','cell'},...
    'VariableNames',{'X','Y'});
cd txt
for i=1:numObservations
    RawData = list(i+2+5340).name;
    RawData = load(RawData);
    vibration = RawData(:,3);
    [S, F, T] = stft(vibration,fs,'Window',kaiser(256,5),'OverlapLength',126,'FFTLength',512);
    VALDATA.X{i} = abs(S);
    VALDATA.Y{i} = Life1(i+5340);
end
cd ../
%%
dataTrain = table('Size',[numObservations 2],...
    'VariableTypes',{'cell','cell'},...
    'VariableNames',{'X','Y'});
dataVal = table('Size',[numObservations 2],...
    'VariableTypes',{'cell','cell'},...
    'VariableNames',{'X','Y'});
for i = 1:height(DATA)
    for j=1:height(DATA.X{1})
        dataTrain.X{i}(j,1) = rms(DATA.X{i}(j,:));
        dataVal.X{i}(j,1) = rms(VALDATA.X{i}(j,:));
    end
    dataTrain.Y{i} = DATA.Y{i};
    dataVal.Y{i} = VALDATA.Y{i};
end
%%
XTrain = table2cell(FeatureData);
YTrain = DATA.Y;
%%
m = min([XTrain{:}],[],2);
M = max([XTrain{:}],[],2);
idxConstant = M == m;

for i = 1:numel(XTrain)
    XTrain{i}(idxConstant,:) = [];
end

numFeatures = size(XTrain{1},1);
%%
mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;
end
%%
thr = 900;
for i = 1:numel(YTrain)
    YTrain{i}(YTrain{i} > thr) = thr;
end
%%
sequence = cell(height(XTrain),1);
for i=1:height(XTrain)
    sequence{i} = cell2mat(XTrain(i,:))';
end
%%

numResponses = size(YTrain{1},1);
numHiddenUnits = 1200;
numFeatures = length(XTrain(1,:));
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 60;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);

net = trainNetwork(sequence,YTrain,layers,options);
%%
XTest = dataTrain.X;
YTest = dataVal.Y;

for i = 1:numel(XTest)
    XTest{i}(idxConstant,:) = [];
    XTest{i} = (XTest{i} - mu) ./ sig;
    YTest{i}(YTest{i} > thr) = thr;
end

YPred = predict(net,XTest,'MiniBatchSize',1);

idx = randperm(numel(YPred),4);
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    
    plot(YTest{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    hold off
    
    ylim([0 thr + 25])
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("RUL")
end
legend(["Test Data" "Predicted"],'Location','southeast')
%%
for i = 1:numel(YTest)
    YTestLast(i) = YTest{i}(end);
    YPredLast(i) = YPred{i}(end);
end
figure
rmse = sqrt(mean((YPredLast - YTestLast).^2))
histogram(YPredLast - YTestLast)
title("RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")
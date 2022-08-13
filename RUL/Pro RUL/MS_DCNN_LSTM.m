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
cd 2nd_test
parfor i=1:numObservations
    RawData = list(i+2).name;
    RawData = load(RawData);
    vibration = RawData(:,1);
    XTrain(i) = rms(vibration);
    YTrain(i) = Life(i);
end
cd ../

%%
layers = [
    sequenceInputLayer(1,"Name","sequence")
    lstmLayer(5000,"Name","lstm_1")
%     lstmLayer(200,"Name","lstm_2")
    fullyConnectedLayer(1,"Name","fc")
    regressionLayer("Name","regressionoutput")];
%% Train the Network
maxEpochs = 20;
miniBatchSize = 32;
%%
options = trainingOptions('adam', ... 
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',20, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',false,...
    'ExecutionEnvironment', 'cpu'); % 'ExecutionEnvironment', 'gpu');
%%
options = trainingOptions('adam', ...
    'MaxEpochs',5, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%%
net = trainNetwork(XTrain,YTrain,layers,options);

YPred = predict(net,XTrain,'MiniBatchSize',miniBatchSize);
figure
plot(t(1:984),YPred,t(1:984),Life)
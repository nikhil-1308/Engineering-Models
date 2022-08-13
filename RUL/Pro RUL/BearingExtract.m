close all;
clear all;
clc;
% 9.4877e+05 RPM 
% given 2000 RPM
% 1.5813e+04 RPS
% dynamic load C = 6.3760e+05
% applied load 6000
% C = 2.9595e+06;
% P = 6000;
% N = 2000;
% LifCycle = ((C/P)^3*10^6)/(6324*N);
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
    [S, F, T] = stft(vibration,fs,'Window',kaiser(256,5),'OverlapLength',126,'FFTLength',512);
%     [S, F, T] = stft(vibration,fs,'Window',kaiser(120,1),'OverlapLength',20,'FFTLength',156);
%     [S, F, T] = stft(vibration,fs,'Window',kaiser(98,1),'OverlapLength',1,'FFTLength',100);
    idx = find(F(:,1)>=-5*(10^3) & F(:,1)<=5*(10^3));
    selectedFeatures = S(idx,:);
    DATA.X{i} = abs(selectedFeatures);
%     DATA.Y{i} = Life(i)*ones(height(S),1);
    DATA.Y{i} = Life(i);
    SpecPlot.STFT{i,1} = S;
    SpecPlot.VIB{i,2} = vibration;
end
cd ../
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
%     [S, F, T] = stft(vibration,fs,'Window',kaiser(120,1),'OverlapLength',20,'FFTLength',156);
%     [S, F, T] = stft(vibration,fs,'Window',kaiser(98,1),'OverlapLength',1,'FFTLength',100);
    idx = find(F(:,1)>=-5*(10^3) & F(:,1)<=5*(10^3));
    selectedFeatures = S(idx,:);
    VALDATA.X{i} = abs(selectedFeatures);
%     DATA.Y{i} = Life(i)*ones(height(S),1);
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
        regData{i}(1,j) = rms(DATA.X{i}(j,:));
        valregData{i}(1,j) = rms(VALDATA.X{i}(j,:));
    end
    dataTrain.Y{i} = DATA.Y{i};
    dataVal.Y{i} = VALDATA.Y{i};
end
%%
newTrainX = vertcat(DATA.X{:});
for r=1:984
    n{r,1} = Life(r)*ones(257,1);
end
newTrainY = vertcat(n{:});
%%
Xtrain = vertcat(regData{:});
Ytrain = cell2mat(dataTrain.Y);
ValX = vertcat(valregData{:});
ValY = dataVal.Y;
[trainedModel, validationRMSE] = trainRegressionModel(Xtrain,Ytrain);
%%
RegPred = trainedModel.predictFcn(Xtrain);
ValRegPred = trainedModel.predictFcn(ValX);
XTR = cell(height(RegPred),1);
XVL = cell(height(RegPred),1);
for s=1:height(RegPred)
    XTR{s} = RegPred(s);
    XVL{s} = ValRegPred(s);
end
%%
XTR = cell(length(y),1);
for s=1:length(y)
    XTR{s} = y(s);
end
%%
figure
stft(SpecPlot.VIB{end,2},fs,'Window',kaiser(256,5),'OverlapLength',126,'FFTLength',512);
% stft(SpecPlot.VIB{end,2},fs,'Window',kaiser(180,1),'OverlapLength',50,'FFTLength',190);
% stft(SpecPlot.VIB{end,2},fs,'Window',kaiser(98,1),'OverlapLength',1,'FFTLength',100);
view(-45,65)
colormap jet
%%
figure
waterfall(F,T,abs(S)')
helperGraphicsOpt(1)
%% CNN REG (CNN_1)
layers = [
    imageInputLayer([257 156 1],"Name","imageinput")
    scalingLayer("Name","scaling","Bias",0)
    convolution2dLayer([8 8],32,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    averagePooling2dLayer([2 2],"Name","avgpool2d_1","Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(1,"Name","fc")
    dropoutLayer(0.2,"Name","dropout")
    regressionLayer("Name","regressionoutput")];
%% CNN_2
layers = [
    imageInputLayer([257 156 1],"Name","imageinput")
    scalingLayer("Name","scaling","Bias",0)
    convolution2dLayer([64 64],32,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    averagePooling2dLayer([2 2],"Name","avgpool2d_1","Padding","same","Stride",[2 2])
    convolution2dLayer([32 32],32,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    averagePooling2dLayer([2 2],"Name","avgpool2d_2","Padding","same","Stride",[2 2])
    convolution2dLayer([16 16],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    convolution2dLayer([8 8],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    dropoutLayer(0.2,"Name","dropout")
    fullyConnectedLayer(1,"Name","fc")
    regressionLayer("Name","regressionoutput")];
%% LSTM REG
layers = [
    sequenceInputLayer(1,"Name","input")
    lstmLayer(1200,"Name","lstm_1")
    lstmLayer(1200,"Name","lstm_2")
%     lstmLayer(4000,"Name","lstm_3")
    fullyConnectedLayer(1,"Name","fc")
    regressionLayer("Name","regressionoutput")];
%% Train the Network
maxEpochs = 120;
miniBatchSize = 32;
%%
options = trainingOptions('adam', ... % adam
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',20, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',false,...
    'ExecutionEnvironment', 'parallel'); % 'ExecutionEnvironment', 'gpu');
%%
validationFrequency = floor(numel(DATA.Y)/miniBatchSize);
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',VALDATA, ...
    'ValidationFrequency',5, ...
    'Plots','training-progress', ...
    'Verbose',false,...
    'ExecutionEnvironment', 'parallel');
%%
Xval = dataVal.X;
Yval = dataVal.Y;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTR,Yval}, ...
    'ValidationFrequency',5, ...
    'Plots','training-progress', ...
    'Verbose',false,...
    'ExecutionEnvironment', 'cpu');
%%
analyzeNetwork(layers)
%% CNN
% net = trainNetwork(dataTrain,layers,options);
net = trainNetwork(DATA,layers,options);
% Net = trainNetwork(DATA,layers,options);
%%
% Pred = predict(Net,DATA,'MiniBatchSize',miniBatchSize);
Pred = predict(net,DATA,'MiniBatchSize',miniBatchSize);
figure
plot(t(1:984),Pred,t(1:984),Life)
%% LSTM
X = dataTrain.X;
Y = dataTrain.Y;
Net = trainNetwork(X,Y,layers,options);
%%
Ytr = dataTrain.Y;
net = trainNetwork(XTR,Ytr,layers,options);
%%
NewPred = predict(net,XTR);
figure
plot(t(1:984),cell2mat(NewPred),t(1:984),Life)
%%
x = RegPred';
tar = Life;
FF = feedforwardnet(20);
FF.trainParam.epochs = 5000;
FF.trainParam.show = 10;
FF.trainParam.goal = 0.1;
FF = train(FF,x,tar);
start = 1;
while start > 0
%     FF = train(FF,x,tar);
    gwb = fpderiv('dperf_dwb',FF,x,tar);
    jwb = fpderiv('de_dwb',FF,x,tar);
    y = FF(x);
    RMSE = sqrt(mse(y-tar));
    error = tar-y;
    perf = perform(FF,tar,y);
    if error(1) <= 0
    	break
    else
        var = tar+jwb(1,:);
        FF.trainFcn = 'trainbr';
        FF = train(FF,y,var);
    end
end
%% Prediction
testData = load('2004.04.18.02.42.55');
TestData = table;
TestData.X = {testData(:,3)};
TestData.Y = 1.0240;

[S, F, T] = stft(TestData.X{1},fs,'Window',kaiser(256,5),'OverlapLength',220,'FFTLength',512);
spectra = abs(vertcat(S(:)));
thrVal = spectra(spectra > mean(spectra)+rms(spectra));
TestData.X = {thrVal(1:Min)};


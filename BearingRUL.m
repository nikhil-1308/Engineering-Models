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
    idx = find(F(:,1)>=-5*(10^3) & F(:,1)<=5*(10^3));
    selectedFeatures = S(idx,:);
    DATA.X{i} = abs(selectedFeatures);
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
    idx = find(F(:,1)>=-5*(10^3) & F(:,1)<=5*(10^3));
    selectedFeatures = S(idx,:);
    VALDATA.X{i} = abs(selectedFeatures);
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
x = RegPred';
tar = Life;

X = tonndata(x,true,false);
T = tonndata(tar,true,false);

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Nonlinear Autoregressive Network with External Input
inputDelays = 1:1;
feedbackDelays = 1:1;
hiddenLayerSize = 5;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

% Prepare the Data for Training and Simulation
[x,xi,ai,t] = preparets(net,X,{},T);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t,xi,ai);
start = 1;
while start > 0
    % Test the Network
    y = net(x,xi,ai);
    y = [cell2mat(y),0];
    error = immse(y,tar);
    if error <= 10
        break
    end
    X1 = tonndata(y,true,false);
    net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
    [y,xi,ai,t] = preparets(net,X1,{},T);
    [net,tr] = train(net,y,t,xi,ai);
end
figure
plot(flip(Life(1:984)),y,flip(Life(1:984)),Life(1:984))
%% Prediction
x = RegPred';
tar = Life;
nets = removedelay(net);
X = tonndata(x,true,false);
T = tonndata(tar,true,false);
[x,xi,ai,t] = preparets(nets,X,{},T);
y1 = nets(x,xi,ai);
y1 = cell2mat(y1);
%%
figure
plot(flip(Life(1:984)),y1,flip(Life(1:984)),Life(1:984))
%%
layers = [
    sequenceInputLayer(1,"Name","sequence")
    lstmLayer(474,"Name","lstm")
    fullyConnectedLayer(1,"Name","fc")
    regressionLayer("Name","regressionoutput")];
%% Train the Network
maxEpochs = 20;
miniBatchSize = 64;
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
    'MaxEpochs',2000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%%
Net = trainNetwork(y1,Life,layers,options);

YPred = predict(Net,y1);
figure
plot(flip(Life(1:984)),YPred,flip(Life(1:984)),Life(1:984))
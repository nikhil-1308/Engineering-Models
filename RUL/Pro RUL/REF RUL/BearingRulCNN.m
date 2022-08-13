%%  nRemaining Useful Life Estimation using Convolutional Neural Network
%% Preprocess Training Data
hsbearing = simulationEnsembleDatastore(fullfile(pwd,'DT'));
hsbearing.SelectedVariables = "Vibration";
ftab = gather(tall(hsbearing));
%%
count = 9;
c=cell(count,1);
for i=1:count
    a=ftab(i,:);
    vib=table2array(a);
    vib=vib{1,1};
%     vib=timetable2table(vib);
%     vib=vib(:,2);
    c{i}=vib;
end
vibration = vertcat(c{:});
vibration = table2array(vibration);
IDnum = numel(vib);
id = 1:count;
IDmat = repmat(id,IDnum,1);
TmStp = numel(vib);
id = reshape(IDmat,[length(vibration),1]);
tp = 1:TmStp;
tp = tp';
timeStamp = repmat(tp,count,1);
Newdata = table(id,timeStamp,vibration);
%% Newdata = table(vibration); {'id','timeStamp','vibration'}
% writetable(Newdata,'Newdata.txt');
writetable(Newdata,'TestNewdata.txt');
%%
filenameTrainPredictors = fullfile(pwd,"Newdata.txt");
rawTrain = localLoadData(filenameTrainPredictors);
%% Examine run-to-failure data for one of the engines.
head(rawTrain.X{1},8)
%% Examine the response data for one of the engines.
rawTrain.Y{1}(1:8)
%% Visualize time-series data for some of the predictors.
stackedplot(rawTrain.X{1},[1,2,3],'XVariable','timeStamp')
%% *Remove Features with Less Variability*
prog = prognosability(rawTrain.X,"timeStamp");
%% 
idxToRemove = prog.Variables==0 | isnan(prog.Variables);
featToRetain = prog.Properties.VariableNames(~idxToRemove);
for i = 1:height(rawTrain)
    rawTrain.X{i} = rawTrain.X{i}{:,featToRetain};
end
%% *Normalize Training Predictors*
[~,Xmu,Xsigma] = zscore(vertcat(rawTrain.X{:}));
preTrain = table();
for i = 1:numel(rawTrain.X)
    preTrain.X{i} = (rawTrain.X{i} - Xmu) ./ Xsigma;
end
%% Clip Responses*
clipResponses = true;
if clipResponses
    rulThreshold = 1010;
    for i = 1:numel(rawTrain.Y)
        preTrain.Y{i} = min(rawTrain.Y{i},rulThreshold);
    end
end
%% *Prepare Data for CNN Training*
WindowLength = 40;
Stride = 1;
dataTrain = localGenerateSequences(preTrain,WindowLength,Stride);
%% *Reshape Sequence Data for imageInputLayer*
numFeatures = size(dataTrain.X{1},2);
InputSize = [WindowLength numFeatures 1];
%%
for i = 1:size(dataTrain,1)
    dataTrain.X{i} = reshape(dataTrain.X{i},InputSize);
end
%%
% gcp;
% n = size(dataTrain,1);
% dataOut = cell(n,1);
% parfor k = 1:n
%     dataOut{k} = HelpReshape(dataTrain,InputSize,k);
% end
% 
% dataTrain.X=dataOut;

% n = numpartitions(ens,gcp);
% n = gcp;
% DT=dataTrain.X;
% parfor ct = 1:n
%     subens = partition(DT,n,ct);
%     while hasdata(subens)
%         datTrain = HelpReshape(dataTrain,InputSize);
%     end
% end
%%
% count = gpuDeviceCount
% gpu = gpuDevice(count)
% 
% X = nndata2gpu(dataTrain.X);
% 
% for i = 1:size(dataTrain,1)
%     X{i} = reshape(X{i},InputSize);
% end
%% Network Architecture
filterSize = [5, 1];
numHiddenUnits = 200;
numResponses = 1;
layers = [
    imageInputLayer(InputSize)
    convolution2dLayer(filterSize,10)
    reluLayer()
    convolution2dLayer(filterSize,20)
    reluLayer()
    convolution2dLayer(filterSize,10)
    reluLayer()
    convolution2dLayer([3 1],5)
    reluLayer()
    fullyConnectedLayer(numHiddenUnits)
    reluLayer()
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer()];
%% Train the Network
maxEpochs = 20;
miniBatchSize = 512;
%%
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.3,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.03, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',0,...
    'ExecutionEnvironment', 'parallel'); % 'ExecutionEnvironment', 'gpu');
%% Train the network using |trainNetwork|.
net = trainNetwork(dataTrain,layers,options);
%% Parallel Training
% net1 = train(net,dataTrain.X,dataTrain.Y,'useParallel','yes','showResources','yes');
% y_pll = net1(dataTrain.X,'useParallel','yes','showResources','yes');
% %% Distributed parallel computation
% pool = gcp;
% net2 = configure(net,dataTrain.X{1},dataTrain.Y{1});
% net2 = train(net2,xdataTrain.X,dataTrain.Y);
% yc = net2(dataTrain.X);
% for i=1:pool.NumWorkers
%   yi = yc{i};
% end
% % y = {yc{:)};
% yd_pll = {yc(:)};
% %% GPU Training
% count = gpuDeviceCount;
% gpu1 = gpuDevice(count);
% % layers.trainFcn = 'trainscg'; % To avoid computation errors by default "trainlm"
% net3 = train(net,dataTrain.X,dataTrain.Y,'useGPU','yes','showResources','yes');
% y_gpu = net3(dataTrain.X,'useGPU','yes','showResources','yes');
% %% Move data to GPU & compute
% Xg = nndata2gpu(dataTrain.X);
% Yg = nndata2gpu(dataTrain.Y);
% % Before training, the network’s tansig layers can be converted to elliotsig layers as follows:
% % numLy = net.numLayers;
% % for i=1:numLy
% %   if strcmp(net.layers{i}.transferFcn,'tansig')
% %     net.layers{i}.transferFcn = 'elliotsig';
% %   end
% % end
% net3 = configure(net,dataTrain.X,dataTrain.Y);  % Configure with MATLAB arrays
% net3 = train(net3,Xg,Yg);    % Execute on GPU with NNET formatted gpuArrays
% Yg = net3(Xg);               % Execute on GPU
% y_OnGpu = gpu2nndata(Yg);          % Transfer array to local workspace
% %% Distributed GPU computation
% % Use all available workers
% net4 = train(net,dataTrain.X,dataTrain.Y,'useParallel','yes','useGPU','only','showResources','yes');
% y_Dgpu = net4(dataTrain.X,'useParallel','yes','useGPU','only','showResources','yes');
% %%
% % Use Selected workers
% net4 = configure(net,dataTrain.X{1},dataTrain.Y{1});
% net4 = train(net4,dataTrain.X,dataTrain.Y,'useGPU','yes','showResources','yes');
% y_slGpu = net4(dataTrain.X,'showResources','yes');
% %%
% % To ensure that the GPUs get used by the first three workers, manually converting each worker’s Composite elements to gpuArrays. 
% % Each worker performs this transformation within a parallel executing spmd block.
% spmd
%   if labindex <= 3
%     Xc = nndata2gpu(dataTrain.X);
%     Yc = nndata2gpu(dataTrain.Y);
%   end
% end
% % Now the data specifies when to use GPUs, so you do not need to tell train and sim to do so.
% net5 = configure(net,Xc{1},Yc{1});
% net5 = train(net5,Xc,Yc,'showResources','yes');
% y_sl2Gpu = net2(Xc,'showResources','yes');
%% Plot the layer graph of the network to visualize the underlying network architecture.
figure;
lgraph = layerGraph(net.Layers);
plot(lgraph)
%% Test the Network
filenameTestPredictors = fullfile(pwd,'Newdata.txt');
filenameTestResponses = fullfile(pwd,'RUL.txt');
dataTest = localLoadData(filenameTestPredictors,filenameTestResponses);
%%
for i = 1:numel(dataTest.X)
    dataTest.X{i} = dataTest.X{i}{:,featToRetain};
    dataTest.X{i} = (dataTest.X{i} - Xmu) ./ Xsigma;
    if clipResponses
        dataTest.Y{i} = min(dataTest.Y{i},rulThreshold);
    end
end
%% 
unitLengths = zeros(numel(dataTest.Y),1);
for i = 1:numel(dataTest.Y)
    unitLengths(i) = numel(dataTest.Y{i,:});
end
dataTest(unitLengths<WindowLength+1,:) = [];
%%
predictions = table('Size',[height(dataTest) 2],'VariableTypes',["cell","cell"],'VariableNames',["Y","YPred"]);
for i=1:height(dataTest)
    unit = localGenerateSequences(dataTest(i,:),WindowLength,Stride);
    predictions.Y{i} = unit.Y;
    predictions.YPred{i} = predict(net,unit,'MiniBatchSize',miniBatchSize);
end
%%
% predictions = table('Size',[height(dataTest) 2],'VariableTypes',["cell","cell"],'VariableNames',["Y","YPred"]);
% lambdaCase = "best";
% for i=1:height(dataTest)
%     unit = localGenerateSequences(dataTest(i,:),WindowLength,Stride);
%     predictions.Y{i} = unit.Y;
%     predictions.YPred{i} = predict(net,unit,'MiniBatchSize',miniBatchSize);
%     predictions.RMSE(i) = sqrt(mean((predictions.Y{i} - predictions.YPred{i}).^2));
%     
%     if isnumeric(lambdaCase)
%         idx = lambdaCase;
%     else
%         switch lambdaCase
%             case {"Random","random","r"}
%                 idx = randperm(height(predictions),1); %Randomly choose a test case to plot
%             case {"Best","best","b"}
%                 idx = find(predictions.RMSE == min(predictions.RMSE)); %Best case
%             case {"Worst","worst","w"}
%                 idx = find(predictions.RMSE == max(predictions.RMSE)); %Worst case
%             case {"Average","average","a"}
%                 err = abs(predictions.RMSE-mean(predictions.RMSE));
%                 idx = find(err==min(err),1);
%         end
%     end
% 
%     actual = predictions.Y{idx};
%     predicted = predictions.YPred{idx};
%     numb = length(actual);
% 
%     if actual ~= 0
%         figure
%         y = zeros(numb,1);
%         yPred = zeros(numb,1);
%         for k=1:numel(actual)
% %             [dpt(k),dt(k)] = localLambdaPlotNew(actual,predicted,k);
%             y(k) = actual(k);
%             yPred(k) = predicted(k);
% %             y = real(k);
% %             yPred = predicted(k);
% %             x = 0:numb(k)-1;
%             x = 0:numel(y)-1;
%             cla
%             hold on
%             plot(x,y,x,yPred)
%             legend("True RUL","Predicted RUL")
%             xlabel("Time stamp (Test data sequence)")
%             ylabel("RUL (Cycles)")
%             % Pause 0.1 seconds to make the animation visible
% %             pause(0.1)
%         end
%     end
% end
%% *Performance Metrics*
for i = 1:size(predictions,1)
    predictions.RMSE(i) = sqrt(mean((predictions.Y{i} - predictions.YPred{i}).^2));
end
%%
figure;
histogram(predictions.RMSE,'NumBins',10);
title("RMSE ( Mean: " + round(mean(predictions.RMSE),2) + " , StDev: " + round(std(predictions.RMSE),2) + " )");
ylabel('Frequency');
xlabel('RMSE');
%% 
figure;
localLambdaPlot(predictions,"best");
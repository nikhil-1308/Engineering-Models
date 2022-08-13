%% Deep Learning Using Bayesian Optimization
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
%     [S, F, T] = stft(vibration,fs,'Window',kaiser(157,1),'OverlapLength',55,'FFTLength',246);
    [S, F, T] = stft(vibration,fs,'Window',kaiser(120,1),'OverlapLength',20,'FFTLength',156);
%     [S, F, T] = stft(vibration,fs,'Window',kaiser(98,1),'OverlapLength',1,'FFTLength',100);
    DATA.X{i} = abs(S);
%     DATA.Y{i} = Life(i)*ones(height(S),1);
    DATA.Y{i} = Life(i);
    SpecPlot.STFT{i,1} = S;
    SpecPlot.VIB{i,2} = vibration;
    spectra = abs(vertcat(S(:)));
    thrVal{i} = spectra(spectra > mean(spectra)+rms(spectra));
end
cd ../
%%
XTrain = DATA.X;
YTrain = DATA.Y;
XValidation = DATA.X;
YValidation = DATA.Y;
%%
optimVars = [
    optimizableVariable('SectionDepth',[1 3],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
    optimizableVariable('Momentum',[0.8 0.98])
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];
%% Perform Bayesian Optimization
ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation);
%%
BayesObject = bayesopt(ObjFcn,optimVars, ...
    'MaxTime',14*60*60, ...
    'IsObjectiveDeterministic',false, ...
    'UseParallel',false);
%% Evaluate Final Network
bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError;
%% 
[YPredicted,probs] = classify(savedStruct.trainedNet,XTest);
testError = 1 - mean(YPredicted == YTest)

NTest = numel(YTest);
testErrorSE = sqrt(testError*(1-testError)/NTest);
testError95CI = [testError - 1.96*testErrorSE, testError + 1.96*testErrorSE]
%% 
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YTest,YPredicted);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
%%
% 
%   figure
%   idx = randperm(numel(YTest),9);
%   for i = 1:numel(idx)
%       subplot(3,3,i)
%       imshow(XTest(:,:,:,idx(i)));
%       prob = num2str(100*max(probs(idx(i),:)),3);
%       predClass = char(YPredicted(idx(i)));
%       label = [predClass,', ',prob,'%'];
%       title(label)
%   end
%
%% 
function ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
%% 
% Define the convolutional neural network architecture.
%% 
% * Add padding to the convolutional layers so that the spatial output size 
% is always the same as the input size.
% * Each time you down-sample the spatial dimensions by a factor of two using 
% max pooling layers, increase the number of filters by a factor of two. Doing 
% so ensures that the amount of computation required in each convolutional layer 
% is roughly the same.
% * Choose the number of filters proportional to |1/sqrt(SectionDepth)|, so 
% that networks of different depths have roughly the same number of parameters 
% and require about the same amount of computation per iteration. To increase 
% the number of network parameters and the overall network flexibility, increase 
% |numF|. To train even deeper networks, change the range of the |SectionDepth| 
% variable.
% * Use |convBlock(filterSize,numFilters,numConvLayers)| to create a block of 
% |numConvLayers| convolutional layers, each with a specified |filterSize| and 
% |numFilters| filters, and each followed by a batch normalization layer and a 
% ReLU layer. The |convBlock| function is defined at the end of this example.

        imageSize = [156 204 1];
        numClasses = numel(unique(YTrain));
        numF = round(16/sqrt(optVars.SectionDepth));
        layers = [
            imageInputLayer(imageSize)
            
            % The spatial input and output sizes of these convolutional
            % layers are 32-by-32, and the following max pooling layer
            % reduces this to 16-by-16.
            convBlock(3,numF,optVars.SectionDepth)
            maxPooling2dLayer(3,'Stride',2,'Padding','same')
            
            % The spatial input and output sizes of these convolutional
            % layers are 16-by-16, and the following max pooling layer
            % reduces this to 8-by-8.
            convBlock(3,2*numF,optVars.SectionDepth)
            maxPooling2dLayer(3,'Stride',2,'Padding','same')
            
            % The spatial input and output sizes of these convolutional
            % layers are 8-by-8. The global average pooling layer averages
            % over the 8-by-8 inputs, giving an output of size
            % 1-by-1-by-4*initialNumFilters. With a global average
            % pooling layer, the final classification output is only
            % sensitive to the total amount of each feature present in the
            % input image, but insensitive to the spatial positions of the
            % features.
            convBlock(3,4*numF,optVars.SectionDepth)
            averagePooling2dLayer(8)
            
            % Add the fully connected layer and the final softmax and
            % classification layers.
            fullyConnectedLayer(numClasses)
            dropoutLayer(0.5,"Name","dropout")
            regressionLayer("Name","regressionoutput")];
%             softmaxLayer
%             classificationLayer];
%% 
% Specify options for network training. Optimize the initial learning rate, 
% SGD momentum, and L2 regularization strength.
% 
% Specify validation data and choose the |'ValidationFrequency'| value such 
% that |trainNetwork| validates the network once per epoch. Train for a fixed 
% number of epochs and lower the learning rate by a factor of 10 during the last 
% epochs. This reduces the noise of the parameter updates and lets the network 
% parameters settle down closer to a minimum of the loss function.

        miniBatchSize = 256;
        validationFrequency = floor(numel(YTrain)/miniBatchSize);
        options = trainingOptions('sgdm', ...
            'InitialLearnRate',optVars.InitialLearnRate, ...
            'Momentum',optVars.Momentum, ...
            'MaxEpochs',60, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',40, ...
            'LearnRateDropFactor',0.1, ...
            'MiniBatchSize',miniBatchSize, ...
            'L2Regularization',optVars.L2Regularization, ...
            'Shuffle','every-epoch', ...
            'Verbose',false, ...
            'Plots','training-progress', ...
            'ValidationData',{XValidation,YValidation}, ...
            'ValidationFrequency',validationFrequency);
%% 
% Use data augmentation to randomly flip the training images along the vertical 
% axis, and randomly translate them up to four pixels horizontally and vertically. 
% Data augmentation helps prevent the network from overfitting and memorizing 
% the exact details of the training images.

        pixelRange = [-4 4];
        imageAugmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXTranslation',pixelRange, ...
            'RandYTranslation',pixelRange);
        datasource = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);
%% 
% Train the network and plot the training progress during training. Close all 
% training plots after training finishes.

        trainedNet = trainNetwork(datasource,layers,options);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))
%% 
% 
% 
% Evaluate the trained network on the validation set, calculate the predicted 
% image labels, and calculate the error rate on the validation data.

        YPredicted = classify(trainedNet,XValidation);
        valError = 1 - mean(YPredicted == YValidation);
%% 
% Create a file name containing the validation error, and save the network, 
% validation error, and training options to disk. The objective function returns 
% |fileName| as an output argument, and |bayesopt| returns all the file names 
% in |BayesObject.UserDataTrace|. The additional required output argument |cons| 
% specifies constraints among the variables. There are no variable constraints.

        fileName = num2str(valError) + ".mat";
        save(fileName,'trainedNet','valError','options')
        cons = [];
        
    end
end
%% 
% The |convBlock| function creates a block of |numConvLayers| convolutional 
% layers, each with a specified |filterSize| and |numFilters| filters, and each 
% followed by a batch normalization layer and a ReLU layer.

function layers = convBlock(filterSize,numFilters,numConvLayers)
layers = [
    convolution2dLayer(filterSize,numFilters,'Padding','same')
    batchNormalizationLayer
    reluLayer];
layers = repmat(layers,numConvLayers,1);
end
%% 
% _Copyright 2019 The MathWorks, Inc._
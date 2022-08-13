clear all;
clc;
%%
cd data
list=dir('*.csv');   % return the list of csv files from the current folder
cd ..\
mkdir DataRUL
%%
m = cell(10,1);
for i=1:10
  cd data
  m{i}=load(list(i).name);   % put into cell array
  Vibration = m{i};
%   Vibration=table(Vibration);
  cd ..\
  cd DataRUL
  if i <= 9
      save(['data-2021110' num2str(i) 'T1' num2str(i-1) '4536Z' '.mat'],'Vibration');
  else
      save(['data-202111' num2str(i) 'T1' num2str(i-1) '4536Z' '.mat'],'Vibration');
  end
%   'yyyyMMdd''T''HHmmss''Z'''
  cd ..\
end
%% Bearing Faults
p = 0.12;
d = 0.02;
n = 8;
th = 0;
f0 = 25;
fs = 97656;
% t = 0:1/fs:1e-1+9/fs;
bpfo = n*f0/2*(1-d/p*cos(th)); % Ball pass frequency, outer race
%%
timeUnit = 'day';
hsbearing = fileEnsembleDatastore(...
    fullfile('.', 'DataRUL'), ...
    '.mat');
hsbearing.DataVariables = "Vibration";
hsbearing.IndependentVariables = "Date";
hsbearing.SelectedVariables = ["Date", "Vibration"];
hsbearing.ReadFcn = @helperReadData;
hsbearing.WriteToMemberFcn = @helperWriteToHSBearing;
tall(hsbearing)
%% RUL Calculation
% hsbearing = simulationEnsembleDatastore(fullfile(pwd,'DT'));
% hsbearing.SelectedVariables = "Vibration";
% data = read(hsbearing);
%% Envelope Spectrum Analysis to Other Fault Types
% data = read(hsbearing);
% dataOuter = table2array(data.Vibration{1});
% xOuter = dataOuter;
% fsOuter = 10000;
% tOuter = (0:length(xOuter)-1)/fsOuter;
% [pEnvOuter, fEnvOuter, xEnvOuter, tEnvOuter] = envspectrum(xOuter, fsOuter);
% 
% figure
% plot(fEnvOuter, pEnvOuter)
% ncomb = 10;
% helperPlotCombs(ncomb, bpfo)
% xlim([0 1000])
% xlabel('Frequency (Hz)')
% ylabel('Peak Amplitude')
% title('Envelope Spectrum: Outer Race Fault')
% legend('Envelope Spectrum', 'BPFO Harmonics')
%% Data Exploration
reset(hsbearing)
tstart = 0;
figure
hold on
while hasdata(hsbearing)
    data = read(hsbearing);
    v = data.Vibration{1};
%     v = table2array(v);
    t = tstart + (1:length(v))/fs;
    % Downsample the signal to reduce memory usage
%     plot(t, v)
    plot(t(1:10:end), v(1:10:end))
    tstart = t(end);
end
hold off
xlabel('Time (s), 6 second per day, 50 days in total')
ylabel('Acceleration (g)')
%%
hsbearing.DataVariables = ["Vibration", "SpectralKurtosis"];
colors = parula(50);
figure
hold on
reset(hsbearing)
% Hours = 1;
day = 1;
while hasdata(hsbearing)
    data = read(hsbearing);
    data2add = table;
    
    % Get vibration signal and measurement date
    v = data.Vibration{1};
%     v = table2array(v);
    % Compute spectral kurtosis with window size = 128
    wc = 128;
    [SK, F] = pkurtosis(v, fs, wc);
    data2add.SpectralKurtosis = {table(F, SK)};
    
    % Plot the spectral kurtosis
%     plot3(F, Hours*ones(size(F)), SK)% 'Color', colors(numel(F)))
    plot3(F, day*ones(size(F)), SK, 'Color', colors(day, :))
    
    % Write spectral kurtosis values
    writeToLastMemberRead(hsbearing, data2add);
    
    % Increment the number of days
%     Hours = Hours + 1;
    day = day + 1;
end
hold off
xlabel('Frequency (Hz)')
% ylabel('Time')
ylabel('Time (day)')
zlabel('Spectral Kurtosis')
grid on
view(-45, 30)
cbar = colorbar;
ylabel(cbar, 'Fault Severity (0 - healthy, 1 - faulty)')
%% Feature Extraction
hsbearing.DataVariables = [hsbearing.DataVariables; ...
    "Mean"; "Std"; "Skewness"; "Kurtosis"; "Peak2Peak"; ...
    "RMS"; "CrestFactor"; "ShapeFactor"; "ImpulseFactor"; "MarginFactor"; "Energy"; ...
    "SKMean"; "SKStd"; "SKSkewness"; "SKKurtosis"];
%% Compute feature values for each ensemble member.
hsbearing.SelectedVariables = ["Vibration", "SpectralKurtosis"];
reset(hsbearing)
while hasdata(hsbearing)
    data = read(hsbearing);
    v = data.Vibration{1};
%     v = table2array(v);
    SK = data.SpectralKurtosis{1}.SK;
    features = table;
    
    % Time Domain Features
    features.Mean = mean(v);
    features.Std = std(v);
    features.Skewness = skewness(v);
    features.Kurtosis = kurtosis(v);
    features.Peak2Peak = peak2peak(v);
    features.RMS = rms(v);
    features.CrestFactor = max(v)/features.RMS;
    features.ShapeFactor = features.RMS/mean(abs(v));
    features.ImpulseFactor = max(v)/mean(abs(v));
    features.MarginFactor = max(v)/mean(abs(v))^2;
    features.Energy = sum(v.^2);
    
    % Spectral Kurtosis related features
    features.SKMean = mean(SK);
    features.SKStd = std(SK);
    features.SKSkewness = skewness(SK);
    features.SKKurtosis = kurtosis(SK);
    
    % write the derived features to the corresponding file
    writeToLastMemberRead(hsbearing, features);
end
%%
hsbearing.SelectedVariables = ["Date","Mean", "Std", "Skewness", "Kurtosis", "Peak2Peak", ...
    "RMS", "CrestFactor", "ShapeFactor", "ImpulseFactor", "MarginFactor", "Energy", ...
    "SKMean", "SKStd", "SKSkewness", "SKKurtosis"];
%% START RUL
featureTable = gather(tall(hsbearing));
% time = 1:height(featureTable);
% featureTable.Hours = hours(time');
%% Convert the table to timetable so that the time information is always associated with the feature values.
featureTable = table2timetable(featureTable);
%% Feature Postprocessing
variableNames = featureTable.Properties.VariableNames;
featureTableSmooth = varfun(@(x) movmean(x, [5 0]), featureTable);
featureTableSmooth.Properties.VariableNames = variableNames;
%% Here is an example showing the feature before and after smoothing.
figure
hold on
plot(featureTable.Date, featureTable.SKMean)
% plot(featureTable.Hours, featureTable.SKMean)
plot(featureTableSmooth.Date, featureTableSmooth.SKMean)
% plot(featureTableSmooth.Hours, featureTableSmooth.SKMean)
hold off
xlabel('Time (Hours)')
ylabel('Feature Value')
legend('Before smoothing', 'After smoothing')
title('SKMean')
%% Training Data
breaktime = datetime(2021, 11, 10);
% [breaktime, M, S]= hms(minutes(60));
% breaktime = hours(10);
breakpoint = find(featureTableSmooth.Date < breaktime, 1, 'last');
% breakpoint = find(featureTableSmooth.Hours < breaktime, 1, 'last');
trainData = featureTableSmooth(1:breakpoint, :);
%% Feature Importance Ranking
featureImportance = monotonicity(trainData, 'WindowSize', 0);
helperSortedBarPlot(featureImportance, 'Monotonicity');
%% Kurtosis of the signal is the top feature based on the monotonicity. 
trainDataSelected = trainData(:, featureImportance{:,:}>0.3);
featureSelected = featureTableSmooth(:, featureImportance{:,:}>0.3);
% trainDataSelected = trainData(:, featureImportance{:,:}>0.001);
% featureSelected = featureTableSmooth(:, featureImportance{:,:}>0.001);
%% Dimension Reduction and Feature Fusion
% Principal Component Analysis (PCA) is used for dimension reduction and feature 
% fusion in this example. Before performing PCA, it is a good practice to normalize 
% the features into the same scale. Note that PCA coefficients and the mean and 
% standard deviation used in normalization are obtained from training data, and 
% applied to the entire dataset.
meanTrain = mean(trainDataSelected{:,:});
sdTrain = std(trainDataSelected{:,:});
trainDataNormalized = (trainDataSelected{:,:} - meanTrain)./sdTrain;
coef = pca(trainDataNormalized);
%% The mean, standard deviation and PCA coefficients are used to process the 
% entire data set.
PCA1 = (featureSelected{:,:} - meanTrain) ./ sdTrain * coef(:, 1);
PCA2 = (featureSelected{:,:} - meanTrain) ./ sdTrain * coef(:, 2);
%% Visualize the data in the space of the first two principal components.
% timeUnit='Hours';
% timeUnit='Hours';
figure
numData = size(featureTable, 1);
scatter(PCA1, PCA2, [], 1:numData, 'filled')
xlabel('PCA 1')
ylabel('PCA 2')
cbar = colorbar;
ylabel(cbar, ['Time (' timeUnit ')'])
%% The plot indicates that the first principal component is increasing as the 
% machine approaches to failure. Therefore, the first principal component is a 
% promising fused health indicator.

healthIndicator = PCA1;
%% Visualize the health indicator.
figure
plot(featureSelected.Date, healthIndicator, '-o')
% plot(featureSelected.Hours, healthIndicator, '-o')
xlabel('Time')
title('Health Indicator')
%% Fit Exponential Degradation Models for Remaining Useful Life (RUL) Estimation
healthIndicator = healthIndicator - healthIndicator(1);
healthIndicator = abs(healthIndicator);
%%
threshold = healthIndicator(end);
%%
mdl = exponentialDegradationModel(...
    'Theta', 1, ...
    'ThetaVariance', 1e6, ...
    'Beta', 1, ...
    'BetaVariance', 1e6, ...
    'Phi', -1, ...
    'NoiseVariance', (0.1*threshold/(threshold + 1))^2, ...
    'SlopeDetectionLevel', 0.05, ...
    'UseParallel', true);
%% Use |predictRUL| and |update| methods to predict the RUL and update the parameter 
% distribution in real time.
totalDay= length(healthIndicator) - 1;
estRULs = zeros(totalDay, 1);
trueRULs = zeros(totalDay, 1);
CIRULs = zeros(totalDay, 2);
pdfRULs = cell(totalDay, 1);

% Keep records at each iteration
% totalHours = length(healthIndicator) - 1;
% estRULs = zeros(totalHours, 1);
% trueRULs = zeros(totalHours, 1);
% CIRULs = zeros(totalHours, 2);
% pdfRULs = cell(totalHours, 1);

% Create figures and axes for plot updating
figure
ax1 = subplot(2, 1, 1);
ax2 = subplot(2, 1, 2);

% for currentHour = 1:totalHours
for currentDay = 1:totalDay
    
    % Update model parameter posterior distribution
%     update(mdl, [currentHour healthIndicator(currentHour)])
    update(mdl, [currentDay healthIndicator(currentDay)])
    
    % Predict Remaining Useful Life
%     [estRUL, CIRUL, pdfRUL] = predictRUL(mdl, ...
%                                          [currentHour healthIndicator(currentHour)], ...
%                                          threshold);
    
    [estRUL, CIRUL, pdfRUL] = predictRUL(mdl, ...
                                         [currentDay healthIndicator(currentDay)], ...
                                         threshold);
%     trueRUL = totalHours - currentHour + 1;
    
    trueRUL = totalDay - currentDay + 1;
    
    % Updating RUL distribution plot
%     helperPlotTrend(ax1, currentHour, healthIndicator, mdl, threshold, timeUnit);
    helperPlotTrend(ax1, currentDay, healthIndicator, mdl, threshold, timeUnit);
    helperPlotRUL(ax2, trueRUL, estRUL, CIRUL, pdfRUL, timeUnit)
    
    % Keep prediction results
%     estRULs(currentHour) = estRUL;
%     trueRULs(currentHour) = trueRUL;
%     CIRULs(currentHour, :) = CIRUL;
%     pdfRULs{currentHour} = pdfRUL;
    
    estRULs(currentDay) = estRUL;
    trueRULs(currentDay) = trueRUL;
    CIRULs(currentDay, :) = CIRUL;
    pdfRULs{currentDay} = pdfRUL;
    
    % Pause 0.1 seconds to make the animation visible
    pause(0.1)
end

% writematrix(estRULs,'RUL.txt');
%% Performance Analysis
alpha = 0.3;
detectTime = mdl.SlopeDetectionInstant;
prob = helperAlphaLambdaPlot(alpha, trueRULs, estRULs, CIRULs, ...
    pdfRULs, detectTime, breakpoint, timeUnit);
%%
title('\alpha-\lambda Plot')
figure
% t = 1:totalHours;
t = 1:totalDay;
hold on
plot(t, prob)
plot([breakpoint breakpoint], [0 1], 'k-.')
hold off
xlabel(['Time (' timeUnit ')'])
ylabel('Probability')
legend('Probability of predicted RUL within \alpha bound', 'Train-Test Breakpoint')
title(['Probability within \alpha bound, \alpha = ' num2str(alpha*100) '%'])
function DATA = localLoadData(filenamePredictors,varargin)

if isempty(varargin)
    filenameResponses = []; 
else
    filenameResponses = varargin{:};
end

%% Load the text file as a table
rawData = readtable(filenamePredictors);
% rawData = filenamePredictors;

% Add variable names to the table
VarNames = {'id','timeStamp','vibration'};

rawData.Properties.VariableNames = VarNames;

if ~isempty(filenameResponses)
    RULTest = dlmread(filenameResponses);
end

% Split the signals for each unit ID
IDs = rawData{:,1};
% IDs = IDs{1};
% IDs = timetable2table(IDs);
% IDs = 1:numel(IDs(:,2));
nID = unique(IDs);
numObservations = numel(nID);

% initialize a table for storing data
DATA = table('Size',[numObservations 2],...
    'VariableTypes',{'cell','cell'},...
    'VariableNames',{'X','Y'});

for i=1:numObservations
    idx = IDs == nID(i);
    DATA.X{i} = rawData(idx,:);
    if isempty(filenameResponses)
        % calculate RUL from time column for train data
        DATA.Y{i} = flipud(rawData.timeStamp(idx))-1;
    else
        % use RUL values from filenameResponses for test data
        sequenceLength = sum(idx);
        endRUL = RULTest(i);
        DATA.Y{i} = [endRUL+sequenceLength-1:-1:endRUL]'; %#ok<NBRAK> 
    end
end
end
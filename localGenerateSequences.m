%% 
% |*Generate Sequences function*|
% 
% This function generate sequences from time-series data given the |WindowLength| 
% and |Stride|. The Output is a table of sequences as matrices and corresponding 
% RUL values as vectors. 

function seqTable = localGenerateSequences(dataTable,WindowLength,Stride)

getX = @(X) generateSequences(X,WindowLength,Stride);
getY = @(Y) Y(WindowLength:Stride:numel(Y));

seqTable = table;
temp = cellfun(getX,dataTable.X,'UniformOutput',false);
seqTable.X = vertcat(temp{:});
temp = cellfun(getY,dataTable.Y,'UniformOutput',false);
seqTable.Y = vertcat(temp{:});
end
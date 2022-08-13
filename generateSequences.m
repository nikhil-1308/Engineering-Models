% sub-function
function seqCell = generateSequences(tsData,WindowLength,Stride)
% returns a cell array of sequences from time-series data using WindowLength and Stride

% create a function to extract a single sequence given start index
getSeq = @(idx) tsData(1+idx:WindowLength+idx,:);
% determine starting indices for sequences
idxShift = num2cell(0:Stride:size(tsData,1)-WindowLength)';
% extract sequences
seqCell = cellfun(getSeq,idxShift,'UniformOutput',false);
end

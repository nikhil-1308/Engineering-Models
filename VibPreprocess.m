function [Vib,vibSpectrum,vibFrequencies,faultValues] = VibPreprocess(data)
% Helper function to preprocess the logged reciprocating pump data.

% Remove the 1st 0.8 seconds of the flow signal
tMin = seconds(0.8);
Vib = data.Vibration{1};
Vib = Vib(Vib.Time >= tMin,:);
Vib.Time = Vib.Time - Vib.Time(1);

% Ensure the flow is sampled at a uniform sample rate
Vib = retime(Vib,'regular','linear','TimeStep',seconds(1e-3));

% Remove the mean from the flow and compute the flow spectrum
fA = Vib;
fA.Data = fA.Data - mean(fA.Data);
[vibSpectrum,vibFrequencies] = pspectrum(fA,'FrequencyLimits',[2 250]);

% Find the values of the fault variables from the SimulationInput
simin = data.SimulationInput{1};
vars = {simin.Variables.Name};
idx = strcmp(vars,'SDrift');
SensorFault = simin.Variables(idx).Value;
idx = strcmp(vars,'ShaftWear');
ShaftFault = simin.Variables(idx).Value;
idx = strcmp(vars,'BearingFault');
BearingFault = simin.Variables(idx).Value;

% Collect the fault values in a cell array
faultValues = {...
    'SensorFault', SensorFault, ...
    'ShaftFault', ShaftFault, ...
    'BearingFault', BearingFault};
end
%% CODE readARD.m
a=arduino()
clear x
clear y
clear realTime
figure
t=timer;
t.Period=0.1;
t.ExecutionMode='fixedRate';
t.TimerFcn='ARDtimer';
cmp=0;
start(t)
tic
while toc<20
end
stop(t)
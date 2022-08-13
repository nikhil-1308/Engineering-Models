%% CODE readARD.m
close all;
clear;
clc;
load('GaussianNN.mat')
load('FeedForwardNet.mat')
nets = removedelay(net);
a=arduino();
fs = 10;
% clear x
% clear y
clear realTime
figure
str = 1;
itr = 1;
init = 0;
cmp=0;
while str > 0
t=timer;
t.Period=1/fs;
t.ExecutionMode='fixedRate';
t.TimerFcn='ARDtimer_1';
start(t)
time = tic;
    while toc(time)<20
    end
stop(t)
tim = init + toc(time);

vibration = normalize(y,'range');
[S, F, TIM] = stft(vibration,fs,'Window',kaiser(20,5),'OverlapLength',10,'FFTLength',512);

subplot(2,1,1)
waterfall(F,TIM,abs(S)')
helperGraphicsOpt(1)

r1 = S(128:255,:);
r2 = S(256:384,:);
selectedFeatures = [r1;r2];
% idx = find(F(:,1)>=-0.1 & F(:,1)<=0.1);
% selectedFeatures = S(idx,:);
absdata = abs(selectedFeatures);
h = height(absdata);
for i=1:h
    v(1,i) = rms(absdata(i,:));
end
vib(itr,:) = v;
pred = trainedModel.predictFcn(vib);

x = pred';
tar(itr) = tim; 
X = tonndata(x,true,false);
T = tonndata(tar,true,false);
[x,xi,ai,t] = preparets(nets,X,{},T);
y1 = nets(x,xi,ai);
y1 = cell2mat(y1);  

init = tim;
% outpred(itr) = y1;
% current(itr) = init;
itr = itr + 1;

subplot(2,1,2)
legend('CurrentLife','PredictedLife')
% plot(current,current,'g',current,outpred,'b--o')
plot(tar,tar,'g',tar,y1,'b--o')
title('condition monitoring')
xlabel('current Life')
ylabel('predicted Life')


if itr >=50
    break
end

end
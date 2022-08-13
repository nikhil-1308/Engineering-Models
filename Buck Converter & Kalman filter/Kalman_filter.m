%Kalman filter design

clc; close all; clear all;
format bank;

%%Continuous time Model

Ac=[-0.026, 0.074, -0.804, -9.809, 0.000;...
    -0.242, -2.017, 73.297, -0.105, -0.001;...
    0.003, -0.153, -2.941, 0.000, 0.000;...
    0.000, 0.000, 1.000, 0.000, 0.000;...
    -0.011, 1.000, 0.000, -75.000, 0.000];
%System is Marginally stable

Bc=[4.594, 0.000;...
    -0.0004, -13.735;...
    0.0002, -24.410;...
    0.000, 0.000;...
    0.000, 0.000];
Cc=eye(5); %5x5 identity mattrix (5 outputs here)..

Dc=zeros(size(Cc,1),size(Bc,2)); %set size automatically

states ={'u' 'w' 'q' '\theta' 'h'};
inputs={'delta_t' 'delta_e'};
outputs=states;

sysc=ss(Ac,Bc,Cc,Dc,'statename',states,...
    'inputname',inputs,...
    'outputname',outputs);

%Discrete time model

%sampling timme
dT=0.05; %(1/0.05) = 20Hz sampling frequency

sys_d=c2d(sysc,dT,'zoh'); %zero-order hold

%obtain your descrete A,B,C,D matrices
Aol=sys_d.A;
Bol=sys_d.B;
Col=sys_d.C;
Dol=sys_d.D;

%% observer design

%parametes G and H
G=Cc; %because 5 states so 5 disturbances (p=n) here
H=zeros(5,5); %%5 outputs so 0 matrix of 5x5

%covariance matrices, process Q. measurement R
Qcov=diag(0.15*ones(1,5)); %Q is 5x5
Rcov=diag(0.05*ones(1,5)); %R is 5x5

%this is your state space with disturbances
sys_kf=ss(Aol,[Bol G],Col,[Dol H],dT);

%obtain L and P, assuming w and v are uncorrelated
%that means N=0 in last parameter
[kest,L,P]=kalman(sys_kf,Qcov,Rcov,0);

%check the value of L that MATLAB returns
%compare L_bar to L (should be equal)
L_bar=(Aol*P*Col)/(Col*P*Col'+Rcov);
Error=norm(abs(L_bar-L));

%assess the stability
Acb=Aol-L*Col;
eig(Acb);





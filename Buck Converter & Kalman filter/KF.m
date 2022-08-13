close all;
clear all;
clc;

N=1000;
dt=0.001; % sampling time
t=dt*(1:N); % time vector

F= [-0.313 56.7 0;
    -0.0139 -0.426 0;
    0 56.7 0];

G=[0.232; 0.0203; 0];

H=[0 0 1];

Q=[0 0 0; 0 0 0; 0 0 0]; % noise matrix

% F=[1 dt;  % A
%     0 1];
% G=[-1/2*dt^2; -dt]; % B
% H=[1 0]; % C
% Q=[0 0; 0 0]; % noise matrix

u=9.80665;
I=eye(3);

% Initialize

y0=100; % position
v0=0; % velocity

% state vector
xt=zeros(3,N);
% xt=zeros(2,N);
% xt(:,1)=[y0;v0]; % first set is int pos & int vel

%% Groud truth

for k=2:N
    xt(:,k)=F*xt(:,k-1)+G*u;
end

R=4;
v=sqrt(R)*randn(1,N); % generate randome noise
z=H*xt+v;

%% Initialise kalman filter

% x=zeros(2,N);
x=zeros(3,N);
% x(:,1)=[10 0];

% P=[50 0;
%     0 0.01];

P=[0.01 0 0; % covariance matrix
    0 0.01 0;
    0 0 0.01];

for k=2:N
    %predict the state vector
    x(:,k)=F*x(:,k-1)+G*u;
    
    %predict the covariance matrix
    P=F*P*F'+Q;
    
    %calculate the kalman gain matrix
    K=P*H'/(H*P*H'+R);
    
    %update/correct the state vector
    x(:,k)=x(:,k)+K*(z(k)-H*x(:,k));
    
    %update the covariance matrix
    P=(I-K*H)*P;
end
    
% plot

figure (1);
subplot(211);
plot(t,z,'g-',t,x(1,:),'b--','Linewidth',2);
hold on;
plot(t,xt(1,:),'r:','Linewidth',1.5);
legend('Measured','Estimated','Ground Truth')
subplot(212);
plot(t,x(2,:),'Linewidth',2);
hold on;
plot(t,xt(2,:),'r:','Linewidth',1.5);

figure(2);
subplot(211);
plot(t,x(1,:)-xt(1,:),'b--','Linewidth',2);
legend('Measured','Estimated','Ground Truth');
subplot(212);
plot(t,x(2,:)-xt(2,:),'Linewidth',2);
legend('Estimated','Ground Truth');





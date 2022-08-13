clear all;
close all;
clc;
% Reference: R. van der Merwe and E. Wan. 
% The Square-Root Unscented Kalman Filter for State and Parameter-Estimation, 2001
%
% By Zhe Hu at City University of Hong Kong, 05/01/2017
n=2;      %dimension of state
m=2;      %dimension of measurement
q=0.1;    %std of process 
r=0.1;    %std of measurement
Qs=q*eye(n); % std matrix of process
Rs=r*eye(m);        % std of measurement  
% f=@(x)[x(1)-0.5*sqrt(x(1));x(2)+0.2*sqrt(x(1))-0.3*sqrt(x(2))];
% h=@(x)[x(2)];
% x=[0;0];
f=@(x)[x(1);x(1)*x(2)];
h=@(x)[x(2)];
x=[0.9416+0.3059*j; 1.1895+0.8642*j];
% f=@(x)[x(2);x(3);0.05*x(1)*(x(2)+x(3))];  % nonlinear state equations
% h=@(x)x(1:m);                               % measurement equation
% x=[0;0;1];                                % initial state
x0=x+q*randn(n,1); %initial state          % initial state with noise
P0 = eye(n);                               % initial square root of state covraiance
N=20;                                     % total dynamic steps
xV = zeros(n,N);          %estmate        % allocate memory
sV = zeros(n,N);          %actual
zV = zeros(m,N);
for k=1:N
  y = h(x) + r*randn(m,1);                     % measurments
  sV(:,k)= x;                             % save actual state
  zV(:,k)  = y;                             % save measurment
  [x0, P0] = UT1(f,x0,P0,h,y,Qs,Rs);            % ukf 
  xV(:,k) = x0;                            % save estimate
  x = f(x) + q*randn(n,1);                % update process 
end

% plot(1:N+1,y,'r')
% hold on;

% for k=1:n                                 % plot results
%   subplot(3,1,k)
%   plot(1:N, sV(k,:), '-', 1:N, xV(k,:), '--')
%   legend('Sys-State','UKF-EST')
% end
% xV;

plot(1:N, sV(2,:), '-', 1:N, xV(2,:), '--')
error=norm(sV(2,:)-xV(2,:))
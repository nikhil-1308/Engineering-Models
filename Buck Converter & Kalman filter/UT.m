function [x0,P0]=UT1(f,x0,P0,h,y,Qs,Rs)
% function [x, P0] = UT1(f,x0,P0,h,y,Qs,Rs);
% SR-UKF   Square Root Unscented Kalman Filter for nonlinear dynamic systems
% [x, S] = ukf(f,x,S,h,z,Qs,Rs) returns state estimate, x and state covariance, P 
% for nonlinear dynamic system (for simplicity, noises are assumed as additive):
%           x_k+1 = f(x_k) + w_k
%           z_k   = h(x_k) + v_k
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
% Inputs:   f: function handle for f(x)
%           x: "a priori" state estimate
%           S: "a priori" estimated the square root of state covariance
%           h: fanction handle for h(x)
%           z: current measurement
%           Qs: process noise standard deviation
%           Rs: measurement noise standard deviation
% Output:   x: "a posteriori" state estimate
%           S: "a posteriori" square root of state covariance
%
% Example:
%{
n=3;      %number of state
q=0.1;    %std of process 
r=0.1;    %std of measurement
Qs=q*eye(n); % std matrix of process
Rs=r;        % std of measurement  
f=@(x)[x(2);x(3);0.05*x(1)*(x(2)+x(3))];  % nonlinear state equations
h=@(x)x(1);                               % measurement equation
s=[0;0;1];                                % initial state
x=s+q*randn(3,1); %initial state          % initial state with noise
S = eye(n);                               % initial square root of state covraiance
N=20;                                     % total dynamic steps
xV = zeros(n,N);          %estmate        % allocate memory
sV = zeros(n,N);          %actual
zV = zeros(1,N);
for k=1:N
  z = h(s) + r*randn;                     % measurments
  sV(:,k)= s;                             % save actual state
  zV(k)  = z;                             % save measurment
  [x, S] = ukf(f,x,S,h,z,Qs,Rs);            % ekf 
  xV(:,k) = x;                            % save estimate
  s = f(s) + q*randn(3,1);                % update process 
end
for k=1:3                                 % plot results
  subplot(3,1,k)
  plot(1:N, sV(k,:), '-', 1:N, xV(k,:), '--')
end
%}
% Reference: R. van der Merwe and E. Wan. 
% The Square-Root Unscented Kalman Filter for State and Parameter-Estimation, 2001
%
% By Zhe Hu at City University of Hong Kong, 05/01/2017
%
n=numel(x0);                                 %numer of states
m=numel(y);                                 %numer of measurements
alpha=1e-3;                                 %default, tunable
ki=0;                                       %default, tunable
beta=2;                                     %default, tunable
lambda=alpha^2*(n+ki)-n;                    %scaling factor
c=n+lambda;                                 %scaling factor
Wm=[lambda/c 0.5/c+zeros(1,2*n)];           %weights for means
Wc=Wm;
Wc(1)=Wc(1)+(1-alpha^2+beta);               %weights for covariance
c=sqrt(c);
X=sigmas(x0,P0,c);                            %sigma points around x
[xhat,Xu,Px,Pk]=ut(f,X,Wm,Wc,n,Qs);          %unscented transformation of process
% X1=sigmas(x1,P1,c);                         %sigma points around x1
% X2=X1-x1(:,ones(1,size(X1,2)));             %deviation of X1
[yhat,Yu,Pyy,Pky]=ut(h,Xu,Wm,Wc,m,Rs);       %unscented transformation of measurments
Pxy=Pk*diag(Wc)*Pky';                        %transformed cross-covariance
K=Pxy/Pyy/Pyy';
x0=xhat+K*(y-yhat);                              %state update
%S=cholupdate(S1,K*P12,'-');                %covariance update
U = K*Pyy';

for i = 1:m
    Px = cholupdate(Px, U(:,i), '-');
end
P0=Px;

function [xhat,Xu,Px,Pk]=ut(f,X,Wm,Wc,n,Rs)
%Unscented Transformation
%Input:

L=size(X,2);
xhat=zeros(n,1);
Xu=zeros(n,L);
for k=1:L                   
    Xu(:,k)=f(X(:,k));       
    xhat=xhat+Wm(k)*Xu(:,k);       
end
Pk=Xu-xhat(:,ones(1,L));
residual=Pk*diag(sqrt(abs(Wc)));
% residual=Y1*diag(sqrt(Wc));                   %It is also right(plural)
[~,Px]=qr([residual(:,2:L) Rs]',0);
if Wc(1)<0
    Px=cholupdate(Px,residual(:,1),'-');
%     Px=Xu*diag(Wc)*Xu'+Rs;
else
    Px=cholupdate(Px,residual(:,1),'+');
% Px=Xu*diag(Wc)*Xu'+Rs;
end
% S=cholupdate(S,residual(:,1));                %It is also right(plural)
%P=Y1*diag(Wc)*Y1'+R;          

function X=sigmas(x,P0,c)
%Sigma points around reference point
%Inputs:

gamma = c*P0';
xhat0 = x(:,ones(1,numel(x)));
X = [x xhat0+gamma xhat0-gamma]; 




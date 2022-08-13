clear all;
close all;
clc;

N=20;
f=@(x)[x(1);x(1)*x(2)];
h=@(x)[x(2)];
x=[0.9416+0.3059*j; 1.1895+0.8642*j];
xp(:,1)=x(:,1);
xu(:,1)=x(:,1);
P=eye(2);
q=0.001;
Q=q*q*eye(2); % Porocess nois
r=0.2;
R=r*r; % Measurement nois
  
%% Plant simulation

for k=1:N
    x(:,k+1)=f(x(:,k));
    y(k+1)=h(x(:,k+1));
end

figure(1)
plot(1:N+1,y,'r')
hold on;

% Adding noise
for k=1:N
    x_noisy(:,k+1)=f(x(:,k))+q*randn;
    y_noisy(k+1)=h(x(:,k+1))+r*randn;
end

figure(1)
plot(1:N+1,y_noisy,'b');
hold on;

%% Extended Kalman Filter

for k=1:N
    % jacobian of f
    F=[1 0; xu(2,k) xu(1,k)];
    
    % jacobian of h
    H=[0 1];
    
    % covariance matrix
    P=F*P*F'+Q;
    S=H*P*H'+R;
    
    % kalman gain
    K=P*H'./S;
    
    % predictions
    xp(:,k+1)=f(xp(:,k)); %stste prediction
    yp(k+1)=h(xp(:,k+1)); %mesurement prediction
    xu(:,k+1)=xp(:,k+1)+K*(y_noisy(k+1)-yp(:,k+1));
    
    % Update
    yu(k+1)=h(xu(:,k+1)); %update measurement
    Pu=(eye(2)-K*H)*P; %update covariance matrix
    P=Pu;
end
    figure(1)
    plot(1:N+1,yu,'--k','Linewidth',2);
    legend('System','NoisySystem','EKF-EST');
    error=norm(x(2,:)-xu(2,:))




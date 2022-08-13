clear all;
close all;
clc;

N=300;
% f=@(x)[x(1)-0.5*sqrt(x(1));x(2)+0.2*sqrt(x(1))-0.3*sqrt(x(2))];
% h=@(x)[x(2)];
% A=@(x)[x(1)-0.5*sqrt(x(1));x(2)+0.2*sqrt(x(1))-0.3*sqrt(x(2))];
% f=@(x)[x(1)-0.5*sqrt(x(1)),0;0.2*sqrt(x(1)),x(2)-0.3*sqrt(x(2))];
% C=@(x)[x(2)];
f=@(x)[x(1)-0.5*sqrt(x(1));x(2)+0.2*sqrt(x(1))-0.3*sqrt(x(2))];
h=@(x)[x(2)];
% B=[0.4;0];
% D=0;
x=[0;0];
x0=[0;0];
xss=[0;0];
uss=[zeros(1,10),ones(1,N-10)];
% xp(:,1)=x(:,1);
% xu(:,1)=x(:,1);
k=0.1;
P=k*eye(1);
q=0.001;
Q=q*q*eye(2); % Porocess nois
Q2=Q;
% r=0.2;
% R=r*r; % Measurement nois
% Q=C'*C;Q2=Q;
R=2*eye(1); R2=R*10;
% nx=2;
% nu=1;
% K0=dlqr(A,B,Q2,R2);
dist=[zeros(1,N/2),-0.1*ones(1,N/2)];
nc=2;runtime=N;

%%%%%%%%%% Initial Conditions 
% nu=size(B,2);ny=size(C,1);nx=size(A,1);
% c = zeros(nu*nc,2); 
% u =zeros(nu,2);  
xin=[x0,x0];
runtime;
J=0;

k3=0.2;k4=0.3;k1=0.5;k2=0.4;
A=[k1,0;k3,0];B=[k2;0];C=[1-k4,0];D=0;ny=65;
[H,P,L,M] = imgpc_predmat(A,B,C,D,ny);
sizey = size(C,1);
R=0.001*eye(1);nu=1;  
[S,X,K,Pr] = imgpc_costfunction(H,P,L,M,R,nu,sizey,ny);
K;
Pr;
Pk=eye(2);
noise = [zeros(1,10),randn(1,N-10)*0.13]; %%% noise

% sizey = size(C,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  SIMULATION
%%% initialise data

xm=x;xp=x;xpr=x;x_noisy=x;xu=x;
yp=0;ym=yp;ypr=yp;yu=yp;
% Prediction Matrices
% Ca=[1;0];Cb=[1;0];
% H=Ca'*Cb;
% E=eye(1);
% K=H'*P;
% lambda=0.9110102;
% Pr=E'*(H'*H+lambda*eye(1))*H';


for i=1:runtime;
    
   r(:,i+1) = uss(:,i);

   %%%%% Update offset 
%    offset(:,i) = ypr(:,i) + noise(:,i)-ym(:,i);   %%% Noise effects measurement only
%    offset(:,i) = yu(:,i)+dist(:,i)-ym(:,i);   %%% Noise effects measurement only
%    offset(:,i) = yu(:,i)+dist(:,i)-r(:,i);   %%% Noise effects measurement only
     offset(:,i) = ypr(:,i) + noise(:,i)-yu(:,i);   %%% Noise effects measurement only
   
   %%%% Update control decisions using a quadratic program
   ufut = -K*xu(:,i) +Pr*(r(:,i+1)-offset(:,i));   %%  (unconstrained)
%    ufut = -0.1*xm(:,i) +1.9*(r(:,i+1)-offset(:,i));   %%  (unconstrained)
%    ufut = -K*xu(:,i) +Pr*(r(:,i+1)-offset(:,i));
    u(:,i) = ufut(1:sizey);
    
    % Simulat input noise
    x_noisy(:,i+1)=f(x_noisy(:,i))+B*u(:,i);%+noise(:,i);
    y_noisy(i+1)=h(x_noisy(:,i+1))+noise(:,i);
    
       % jacobian of f
    F=[1 0; xu(2,i) xu(1,i)];
    
    % jacobian of h
    H=[0 1];
    
    % covariance matrix
    Pk=F*Pk*F'+Q;
    S=H*Pk*H'+R;
    
    % kalman gain
    K1=Pk*H'./S;
    
    % predictions
    xp(:,i+1)=f(xp(:,i))+B*u(:,i); %stste prediction
    yp(i+1)=h(xp(:,i+1)); %mesurement prediction
    xu(:,i+1)=xp(:,i+1)+K1*(y_noisy(i+1)-yp(:,i+1));
    
   % Update
    yu(i+1)=h(xu(:,i+1)); %update measurement
    Pu=(eye(2)-K1*H)*Pk; %update covariance matrix
    Pk=Pu;

   %%%% Simulate model
      xm(:,i+1) = f(xm(:,i))+B*u(:,i);
      ym(:,i+1) = h(xm(:,i+1));
   %%%% Simulate process
      xpr(:,i+1) = f(xpr(:,i))+B*u(:,i);
      ypr(:,i+1) = h(xpr(:,i+1))+dist(:,i);
      
      e(:,i+1)=uss(:,i)-ypr(:,i+1);
      
end

%%%%% Ensure data lengths are all compatible
u(:,i+1)=u(:,i);
% Du(:,i+1)=Du(:,i)*0;
% r(:,i+1) = ref(:,i+1);
% noise = noise(:,1:i+1);
% d = dist(:,1:i+1);

x=xpr;y=ypr;

v=2:length(y);
plot(v,y(v),'b','linewidth',2); title(['SOMPC output for n_c=',num2str(nc)],'fontsize',18)
hold on; 
% plot(v,ym(v),'--k','linewidth',2);
plot(v,yp(v),'--k','linewidth',2);
plot(v,uss,'r','linewidth',2);
plot(v,y_noisy(v),'k','linewidth',1);
plot(v,dist,'g','linewidth',2);
plot(v,u(v),'m','linewidth',2);
plot(v,e(v),'--m','linewidth',1);
legend('Output','target','noisy','disturbance','input','error');



%% Control test 2

for i=1:runtime;

xss(:,i+1)=A(xss(:,i))+B*uss(:,i);
yss(i+1)=C(xss(:,i+1));
    
% xss(:,i)=M1*(ref(:,i+1)-dist(:,i)); %% based on next target, current disturbance
% uss(:,i)=M2*(ref(:,i+1)-dist(:,i)); 
xhat(:,i+1)=x(:,i)-xss(:,i);

%%%%% Control law
% c(:,i) = KK*xhat(:,i);  
uhat(:,i) = -[1,1]*xhat(:,i);
u(:,i)=uhat(:,i)+uss(:,i);

%%%% Simulate model with disturbance      
%      x(:,i+1) = A*x(:,i) + B*u(:,i) ;
%      y(:,i+1) = C*x(:,i+1)+dist(:,i+1);
     
     x(:,i+1)=A(x(:,i))+B*u(:,i)+noise(:,i);
     y(i+1)=C(x(:,i+1))+dist(:,i);

%%% update cost (based on deviation variables)
%      J(i)=xhat(:,i)'*SX*xhat(:,i)+2*c(:,i)'*SXC'*xhat(:,i)+c(:,i)'*SC*c(:,i);
end

%%%% Ensure all variables have conformal lengths
u(:,i+1) = u(:,i);  
% c(:,i+1)=c(:,i);
% J(:,i+1)=J(:,i);
% J(1)=J(2);

v=2:length(y);
plot(v,y(v),'b','linewidth',2); title(['SOMPC output for n_c=',num2str(nc)],'fontsize',18)
hold on; 
plot(v,yss(v),'--k','linewidth',2);
plot(v,uss,'r','linewidth',2);
% plot(v,noise(v),'--k','linewidth',2);
plot(v,dist,'g','linewidth',2);
plot(v,u(v),'m','linewidth',2);
legend('Output','SS','target','disturbance','input');
% legend('Output','target','noise','disturbance','input');



%% Initialization

for k=1:N
    x(:,k+1)=f(x(:,k))+B*u(:,k);
    y(k+1)=h(x(:,k+1));
end

figure(1)
plot(1:N+1,y,'r')
hold on;

% Adding noise
for k=1:N
    x_noisy(:,k+1)=f(x(:,k))+B*u(:,k)+q*randn;
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
    xp(:,k+1)=f(xp(:,k))+B*u(:,k); %stste prediction
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
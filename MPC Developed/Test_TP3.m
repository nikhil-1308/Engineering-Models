clear all;
close all;
clc;

N=300;
f=@(x)[x(1)-0.5*sqrt(x(1));x(2)+0.2*sqrt(x(1))-0.3*sqrt(x(2))];
h=@(x)[x(2)];
x=[0;0];
% x0=[0;0];
xss=[0;0];
ref=[zeros(1,10),ones(1,N-10)];
% xp(:,1)=x(:,1);
% xu(:,1)=x(:,1);
% k=0.1;
% P=k*eye(1);
q=0.001;
Q=q*q*eye(2); % Porocess nois
% Q2=Q;
% r=0.2;
% R=r*r; % Measurement nois
% Q=C'*C;Q2=Q;
% R=2*eye(1); R2=R*10;
% nx=2;
% nu=1;
% K0=dlqr(A,B,Q2,R2);
dist=[zeros(1,N/2),-0.1*ones(1,N/2)];
nc=2;runtime=N;

%%%%% constraints
umin=[-1;-2] ;    %%% umin<u<umax
umax=[1;0.5];
Kxmax=[1 0.2;-0.1 0.4;-1,-0.2;0.1,-0.4;0 0;0 0];
xmax=[1.5;1;0.5;0];

% umin=[-1;-2] ;    %%% umin<u<umax
% umax=[2;1.5];
% Kxmax=[1 0.2 0;-0.1 0.4 0;-1,-0.2 0;0.1,-0.4 0;0 0 1;0 0 -1];
% xmax=[4;4;2.5;0.5;2;2];

%%%%%%%%%% Initial Conditions 
% nu=size(B,2);ny=size(C,1);nx=size(A,1);
% c = zeros(nu*nc,2); 
% u =zeros(nu,2);  
% J=0;
runtime;

k3=0.2;k4=0.3;k1=0.5;k2=0.4;
A=[-k1,0;1-k4+k3,0];B=[k2;0];C=[0,1];D=0; 
% A=[0.7555 0.25;-0.1991 0];B=[-0.5;0];C=[0 1];D=0;
nx=size(A,1);
ny1=size(C,1);
ny=9;nu=1;

[H,P,Q] = mpc_predmat(A,B,ny);

[H,P,L,M] = imgpc_predmat(A,B,C,D,ny);
sizey = size(C,1);
R=0.1*eye(1);  
[S,X,K,Pr] = imgpc_costfunction(H,P,L,M,R,nu,sizey,ny);
K;
Pr;
Pk=eye(2);
noise = [zeros(1,10),randn(1,N-10)*0.13]; %%% noise
Q=C'*C;Q2=Q;
R2=R*10;

%%%% Control law parameters
[SX,SC,SXC,Spsi,K,Psi,Kz]=chap4_suboptcost(A,B,C,D,Q,R,Q2,R2,nc);
SC=(SC+SC')/2; %%% to avoid silly error messages
if norm(SXC)<1e-10; SXC=SXC*0;end
KK=inv(SC)*SXC';
Ksompc=[K+KK(1:nu,:)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  SIMULATION
%%% initialise data

xm=x;xp=x;xpr=x;x_noisy=x;xu=x;
yp=0;ym=yp;ypr=yp;yu=yp;
u=zeros(1,N);

%%%% Estimate steady-state values
%%%  [xss;uss]=[M1;M2](r-d)
M=inv([C,zeros(ny1,nu);A-eye(nx),B]);
M1=M(1:nx,1:ny1);
M2=M(nx+1:nx+ny1,1:ny1);

%%%%% For now assume that output targets are all 1
M1=M1*ones(nu,1);
M2=M2*ones(nu,1);

%%%%%  Find MAS as M x + N cfut <= f
G=[-Kz;Kz;[Kxmax,zeros(size(Kxmax,1),nc*nu)]];
% G=[-Kz;Kz;[Kxmax,zeros(size(Kxmax,1),1)]];
f=[umax;-umin;xmax]; 
Kss=[-M2;M2;-Kxmax*M1];
f=f+Kss;
if any(f<0);
    disp('Likely error or infeasibility in desired target');
end

[F,t]=findmas(Psi,G,f);
N=F(:,nx+1:nx+nu*nc);
M=F(:,1:nx);

%%%%% Settings for quadratic program
opt = optimset('quadprog');
opt.Diagnostics='off';    %%%%% Switches of unwanted MATLAB displays
opt.LargeScale='off';     %%%%% However no warning of infeasibility
opt.Display='off';
opt.Algorithm='active-set';



for i=1:runtime;
    
   xss(:,i)=M1; %% aSSUMES R=1 
   uss(:,i)=M2;
   
   xhat(:,i)=xu(:,i)-xss(:,i);
    
   r(:,i+1) = ref(:,i);

   %%%%% Update offset 
%    offset(:,i) = ypr(:,i) + noise(:,i)-ym(:,i);   %%% Noise effects measurement only
     offset(:,i) = ypr(:,i) + noise(:,i)-yu(:,i);   %%% Noise effects measurement only
   
   %%%% Update control decisions using a quadratic program
     ufut = -K*xu(:,i) +Pr*(r(:,i+1)-offset(:,i));   %%  (unconstrained)
%    ufut = -0.1*xm(:,i) +1.9*(r(:,i+1)-offset(:,i));   %%  (unconstrained)
    
%     xhat(:,i)=xpr(:,i)-xu(:,i);

    [cfut,vv,exitflag] = fmincon(@Output, [real(xhat(1,i)) real(xhat(2,i))], [], [], [], [], [0 0], [],@mycons);
    
    if exitflag==-2;disp('No feasible solution');
    cfut=cfast(:,i);
    %else; disp('looping')    
    end
    
    c(:,i)=cfut;
    uhat(:,i)=-K*xhat(:,i)+c(1:nu,i);%+Pr*(r(:,i+1)-offset(:,i));
    u(:,i)=uhat(:,i);%+r(:,i);
    u(:,i)=uhat(:,i)+uss(:,i);
%     u(:,i)=ufut(:,i)*c(1:nu,i);
    
%     u(:,i) = u(1:sizey);
    
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

plot(v,y(v),'b','linewidth',2);
hold on; 
% plot(v,ym(v),'--k','linewidth',2);
plot(v,yp(v),'--k','linewidth',2);
plot(v,uss,'r','linewidth',2);
plot(v,y_noisy(v),'k','linewidth',1);
plot(v,dist,'g','linewidth',2);
plot(v,u(v),'m','linewidth',2);
plot(v,e(v),'--m','linewidth',1);
legend('Output','target','noisy','disturbance','input','error');


function Y=Output(x)
    Y=x(2)+0.2*sqrt(x(1))-0.3*sqrt(x(2));
end
function S=States(x)
    S=x(1)-0.5*sqrt(x(1))+0.4;
end
function [c,ceq]=mycons(x)
c=States(x)-1;
ceq=[];
end
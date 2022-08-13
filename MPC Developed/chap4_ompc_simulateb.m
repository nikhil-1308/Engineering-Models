%%% Simulation of dual mode optimal predictive control with tracking
%%%
%%  [J,x,y,u,c,KSOMPC] = chap4_ompc_simulateb(A,B,C,D,nc,Q,R,Q2,R2,x0,runtime,ref,dist)
%%
%%   Q, R denote the weights in the actual cost function
%%   Q2, R2 are the weights used to find the terminal mode LQR feedback
%%   nc is the control horizon
%%   A, B,C,D are the state space model parameters
%%   x0 is the initial condition for the simulation
%%   J is the predicted cost at each sample
%%   c is the optimised perturbation at each sample
%%   x,y,u are states, outputs and inputs
%%   KSOMPC unconstrained feedback law
%%   ref a target signal
%%   output disturbance signal (assumes known for simplicity)

function [J,x,y,u,c,Ksompc] = chap4_ompc_simulateb(A,B,C,D,nc,Q,R,Q2,R2,x0,runtime,ref,dist,noise)

%%%%%%%%%% Initial Conditions 
nu=size(B,2);ny=size(C,1);nx=size(A,1);
c = zeros(nu*nc,2); u =zeros(nu,2);  
x=[x0,x0];z=x;xe=x; xp=x;
y=C*x;
z=[x;y];
de=y;
runtime;
J=0;

% [K,L,Ao,Bo,Co,Do,Kd,Pr,Mx,Mu] = ompc_observor(A,B,C,D,Q2,R2);

%%%%%   The optimal predicted cost at any point 
%%%%%     J = c'*SC*c + 2*c'*SCX*xhat + x'*Sx*xhat
%%%%%   xhat =x-xss
%%%% Control law parameters
[SX,SC,SXC,Spsi,K]=chap4_suboptcost(A,B,C,D,Q,R,Q2,R2,nc);
if norm(SXC)<1e-10; SXC=SXC*0;end
KK=inv(SC)*SXC';
Ksompc=[K+KK(1:nu,:)];

%%%% Estimate steady-state values
%%%  [xss;uss]=[M1;M2](r-d)
M=inv([C,zeros(ny,nu);A-eye(nx),B]);
M1=M(1:nx,1:ny);
M2=M(nx+1:nx+ny,1:ny);

Kds = Ksompc*M1+M2;
Prs=Kds;

P=Prs*eye(nx);
% P=0.1*eye(nx);
I=eye(nx);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   SIMULATION

for i=2:runtime;

% xss(:,i)=M1*(ref(:,i+1)-dist(:,i)); %% based on next target, current disturbance
% uss(:,i)=M2*(ref(:,i+1)-dist(:,i)); 
% xhat(:,i)=xp(:,i)-xss(:,i);
% 
% c(:,i) = KK*xhat(:,i); 
% 
% xss(:,i)=M1*(ref(:,i+1)-de(:,i)); %% based on next target, current disturbance
% uss(:,i)=M2*(ref(:,i+1)-de(:,i)); 
% xhat(:,i)=z(:,i)-xss(:,i);  %%% Use state of independent model
% 
% % %%% Control law
% c(:,i) = KK*xhat(:,i);  
% uhat(:,i) = -Ksompc*xhat(:,i);
% u(:,i)=uhat(:,i)+uss(:,i);

% %%% Control law
% c(:,i) = KK*xe(:,i);  
% uhat(:,i) = -Ksompc*xe(:,i);
% u(:,i)=uhat(:,i)+uss(:,i);



    %%% collect state and disturbance estimates
%     xe(:,i)=z(1:nx,i);
%     de(:,i)=z(nx+1:end,i);


    %%%% Control law
%      u(:,i) = -Ksompc*xe(:,i) - Kds* de(:,i) + Prs*ref(:,i+1);
      u(:,i) = -Ksompc*xe(:,i) - Kds* dist(:,i) + Prs*ref(:,i+1);
%      u(:,i) = (-Ksompc*xe(:,i) - Kds* (dist(:,i)) + Prs*ref(:,i+1))*noise(:,i);

% % % %%%% Simulate model with disturbance      
%     z(:,i+1) = A*z(:,i) + B*u(:,i)+noise(:,i) ;
%     ym(:,i+1) = C*z(:,i+1);
      
    % % %%%% Simulate model with Noise      
    x(:,i+1) = A*x(:,i) + B*ref(:,i)+noise(:,i) ;
    ym(:,i+1) = C*x(:,i+1);
     
    
    %predict the state vector
%     xe(:,i+1)=A*xe(:,i)+B*ref(:,i);
    xe(:,i+1)=A*xp(:,i)+B*u(:,i);
    
    %predict the covariance matrix
    P=A*P*A'+Q;
    
    %calculate the kalman gain matrix
    K1=P*C'/(C*P*C'+R);
    
    %update/correct the state vector
%     xe(:,i+1)=xe(:,i)+K1*(de(i)-C*xe(:,i));
    xe(:,i+1)=xe(:,i+1)+K1*(z(i)-C*xe(:,i+1));
    
    %update the covariance matrix
    P=(I-K1*C)*P;
    
%%%% Simulate model with disturbance      
     xp(:,i+1) = A*xp(:,i) + B*u(:,i);
     y(:,i+1) = C*xp(:,i+1)+dist(:,i+1);
     
     
 %%% Observer part
%         z(:,i+1) = Ao*z(:,i) +Bo*u(:,i) + L*(y(:,i) - Co*z(:,i)+noise(:,i));
%      
%         z(:,i+1)=A*z(:,i) +B*u(:,i) + L*(y(:,i) - C*z(:,i)+noise(:,i));
     
     
       %%%% Estimate disturbance
%         de(:,i+1)=y(:,i+1)-ym(:,i+1)+noise(:,i);
%         de(:,i+1)=y(:,i+1)+ym(:,i+1)+noise(:,i);
        z(:,i+1)=y(:,i+1)+noise(:,i)+ym(:,i+1);
%        z(:,i+1)=y(:,i+1)+L*(ym(:,i+1)-Co*z(:,i)+noise(:,i));

   
% xss(:,i)=M1*(ref(:,i+1)-de(:,i)); %% based on next target, current disturbance
% uss(:,i)=M2*(ref(:,i+1)-de(:,i)); 
% xhat(:,i)=x(:,i)-xss(:,i);
% c(:,i) = KK*xhat(:,i);

xss(:,i)=M1*(ref(:,i+1)-dist(:,i)); %% based on next target, current disturbance
uss(:,i)=M2*(ref(:,i+1)-dist(:,i)); 
xhat(:,i)=xp(:,i)-xss(:,i);
c(:,i) = KK*xhat(:,i);

%%% update cost (based on deviation variables)
     J(i)=xhat(:,i)'*SX*xhat(:,i)+2*c(:,i)'*SXC'*xhat(:,i)+c(:,i)'*SC*c(:,i);
%      J(i)=xe(:,i)'*SX*xe(:,i)+2*c(:,i)'*SXC'*xe(:,i)+c(:,i)'*SC*c(:,i);
end

%%%% Ensure all variables have conformal lengths
u(:,i+1) = u(:,i);  
c(:,i+1)=c(:,i);
J(:,i+1)=J(:,i);
J(1)=J(2);



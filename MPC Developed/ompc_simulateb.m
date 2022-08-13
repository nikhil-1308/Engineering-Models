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

function [J,x,y,u,c,Ksompc] = chap4_ompc_simulateb(A,B,C,D,nc,Q,R,Q2,R2,x0,runtime,ref,dist)

%%%%%%%%%% Initial Conditions 
nu=size(B,2);ny=size(C,1);nx=size(A,1);
% ny=size(C,1);
c = zeros(nu*nc,2); u =zeros(nu,2);  
x=[x0,x0];
y=C*x;
runtime;
J=0;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   SIMULATION

for i=2:runtime;

xss(:,i)=M1*(ref(:,i+1)-dist(:,i)); %% based on next target, current disturbance
uss(:,i)=M2*(ref(:,i+1)-dist(:,i)); 
xhat(:,i)=x(:,i);%-xss(:,i);

%%%%% Control law
c(:,i) = KK*xhat(:,i);  
uhat(:,i) = -Ksompc*xhat(:,i);
u(:,i)=uhat(:,i);%+uss(:,i);
% u(:,i)=ref(:,i);%+uss(:,i);

%%%% Simulate model with disturbance      
     x(:,i+1) = A*x(:,i) + B*u(:,i) ;
     y(:,i+1) = C*x(:,i+1)+dist(:,i+1);

%%% update cost (based on deviation variables)
     J(i)=xhat(:,i)'*SX*xhat(:,i)+2*c(:,i)'*SXC'*xhat(:,i)+c(:,i)'*SC*c(:,i);
end

%%%% Ensure all variables have conformal lengths
u(:,i+1) = u(:,i);  
c(:,i+1)=c(:,i);
J(:,i+1)=J(:,i);
J(1)=J(2);



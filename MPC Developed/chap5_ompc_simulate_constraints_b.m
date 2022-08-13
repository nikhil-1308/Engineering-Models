%%% Simulation of dual mode optimal predictive control
%%%
%%  [J,x,y,u,c,KSOMPC] = chap5_ompc_simulate_constraintsb(A,B,C,D,nc,Q,R,Q2,R2,x0,runtime,umin,umax,Kxmax,xmax,rdmin,rdmax)
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
%%
%%   Assumes a target signal of unity
%%   output disturbance signal (assumes ZERO for simplicity)

%%  Adds in constraint handling with constraints
%%  umin<u < umax   and    Kxmax*x < xmax
%%
%% Care must be taken to ensure the problem given is feasible

function [J,x,y,u,c,Ksompc,F,t,M1] = chap5_ompc_simulate_constraintsb(A,B,C,D,nc,Q,R,Q2,R2,x0,runtime,ref,dist,noise,umin,umax,Kxmax,xmax,rdmax,rdmin)

%%%%%%%%%% Initial Conditions 
nu=size(B,2);
nx=size(A,1);
ny=size(C,1);
c = zeros(nu*nc,2); u =zeros(nu,2);  
x=[x0,x0]; xe=x; xp=x;xm=x;xt=x;
y=C*x;
ye=C*x;
% z=[x;y];
de=y;
z=y;
runtime;
J=0;

%%%%% The optimal predicted cost at any point 
%%%%%     J = c'*SC*c + 2*c'*SCX*x + x'*Sx*x
%%%%  Builds an autonomous model Z= Psi Z, u = -Kz Z  Z=[x;cfut];
%%%%
%%%% Control law parameters
[SX,SC,SXC,Spsi,K,Psi,Kz]=chap4_suboptcost(A,B,C,D,Q,R,Q2,R2,nc);
SC=(SC+SC')/2; %%% to avoid silly error messages
if norm(SXC)<1e-10; SXC=SXC*0;end
KK=inv(SC)*SXC';
Ksompc=[K+KK(1:nu,:)];

%%%% Estimate steady-state values
%%%  [xss;uss]=[M1;M2](r-d)
M=inv([C,zeros(ny,nu);A-eye(nx),B]);
M1=M(1:nx,1:ny);
M2=M(nx+1:nx+ny,1:ny);

%%%%% For now assume that output targets are all 1
M1=M1*ones(nu,1);
M2=M2*ones(nu,1);

Kds = Ksompc*M1+M2;
Prs=Kds;

P=Prs(2,1)*eye(nx);

I=eye(nx);

%%%%% Define constraint matrices using invariant set methods on
%%%%%   Z= Psi Z  
%%%%%   uhat=-Kz Z  umin<uhat+uss<umax   
%%%%%   Kxmax *(xhat+xss) <xmax
%%%%%
%%%%% First define constraints at each sample as G*x<f
%%%%%
%%%%%  Find MAS as M x + N cfut <= f
G=[-Kz;Kz;[Kxmax,zeros(size(Kxmax,1),nc*nu)]];
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   SIMULATION

for i=2:runtime;


xss(:,i)=M1; %% aSSUMES R=1 
uss(:,i)=M2;
% xss(:,i)=M1*(ref(:,i+1)-(dist(:,i))); %% based on next target, current disturbance
% uss(:,i)=M2*(ref(:,i+1)-dist(:,i)); 
xhat(:,i)=xe(:,i)-xss(:,i);
% xhat(:,i+1)=xm(:,i)-xss(:,i);

% %%%%% Unconstrained control law
cfast(:,i) = KK*xhat(:,i);  
% uhatfast(:,i) = -Ksompc*xhat(:,i);
% ufast(:,i)=uhatfast(:,i)+uss(:,i);


%%%% constrained control law
%%%%  N c + M xhat +V(r-d) <=t
%%%%  J = c'*SC*c + 2*c'*SCX*xhat 
[cfut,vv,exitflag] =        quadprog(SC,SXC'*xhat(:,i),N,t-M*xhat(:,i), [], [], [], [],[], x, opt);
%[X,FVAL,EXITFLAG,OUTPUT] = quadprog(H,       obj,     A,       b,     Aeq, beq,lb, ub,  , x0,opt)
                                  % 'H' is problem matrix
if exitflag==-2;disp('No feasible solution');
    cfut=cfast(:,i);
%else; disp('looping')
    
end

c(:,i)=cfut;
uhat(:,i)=-K*xhat(:,i)+c(1:nu,i);
u(:,i)=uhat(:,i)+uss(:,i);
      
    % % %%%% Simulate model with Noise      
    xm(:,i+1) = A*x(:,i) + B*u(:,i)+noise(:,i) ;
    ym(:,i+1) = C*xm(:,i+1);
     
    
    %predict the state vector
    xe(:,i)=A*xe(:,i-1)+B*u(:,i);
    
    %predict the covariance matrix
    P=A*P*A'+Q;
    
    %calculate the kalman gain matrix
    K1=P*C'/(C*P*C'+R);
    
    %update/correct the state vector
    xe(:,i+1)=xe(:,i)+K1*(z(i)-C*xe(:,i));
    
    %update the covariance matrix
    P=(I-K1*C)*P;

%%%% Simulate model with estimates      
     xe(:,i+1) = A*xe(:,i) + B*u(:,i);
     ye(:,i+1) = C*xe(:,i+1)+dist(:,i+1);
     
%%%% Simulate model with disturbance      
     x(:,i+1) = A*x(:,i) + B*u(:,i);
     y(:,i+1) = C*x(:,i+1)+dist(:,i+1);
     
     
%%%% Estimate disturbance
     z(:,i+1)=y(:,i+1)+noise(:,i)+ym(:,i+1);

%%% update cost (based on deviation variables)
     J(i)=xhat(:,i)'*SX*xhat(:,i)+2*c(:,i)'*SXC'*xhat(:,i)+c(:,i)'*SC*c(:,i);

end
%%%% Ensure all variables have conformal lengths
u(:,i+1) = u(:,i);  
c(:,i+1)=c(:,i);
J(:,i+1)=J(:,i);
J(1)=J(2);



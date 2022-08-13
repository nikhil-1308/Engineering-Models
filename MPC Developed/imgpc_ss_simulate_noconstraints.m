%%% Simulation of independent model GPC with a state space model (Figure 1 on)
%%%
%%% Uses the cost function   J = sum (r-y)^2 + R(u-uss)^2
%%% (weights absolute inputs not increments)
%%%
%%  [x,y,u,r] = imgpc_simulate(A,B,C,D,R,ny,nu,umax,umin,Dumax,x0,ref,dist,noise);
%%
%%%     x(k+1) = A x(k) + B u(k)              x0 is the initial condition
%%%     y(k)   = C x(k) + D u(k) + dist       Note: Assumes D=0, dist unknown
%%%
%%  input constraint         umax, umin
%%  input rate constraint    Dumax
%%  reference trajectory     ref, r
%%  steady-state input       uss
%%  Output/input horizons    ny,nu
%%  Weighting matrix in J    R
%%  measurement noise        noise
%%  
%% Author: J.A. Rossiter  (email: J.A.Rossiter@shef.ac.uk)


function [x,y,u,r] = imgpc_simulate(A,B,C,D,R,ny,nu,x0,ref,dist,noise);


sizey = size(C,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Find GPC law  and observor (using independent model approach)   
%%%%%  Control law is written as  a combination of feedback and
%%%%%  simulation of the independent model
%%%%%
%%%%%         u = -Kz + Pr (r - offset);     offset = yprocess-ymodel
%%%%%  
%%%%%        Independent model is 
%%%%%        z = Az + Bu;  y = Cz;      
%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%  Predictions are yfut = P*x + H*ufut + L*offset
%%%  Steady state input is estimated as M*(r-offset)
%%%  Note: As using inputs (not increments) require all columns of H
[H,P,L,M] = imgpc_predmat(A,B,C,D,ny);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Find control law minimising  J = sum yfut'yfut + ufut*R*ufut
%%%  Assume that u(k+nu+i) = u(k+nu-1) in predictions
%%%  The optimal cost is 
%%%     J = ufut'Sufut + ufut'X*[x;r-offset] + unconstrained optimal
%%%  The control law is
%%%     ufut = -K*x +Pr*(r-offset)
%%%
%%%  Predictions are yfut = P*x + H*ufut + L*offset
[S,X,K,Pr] = imgpc_costfunction(H,P,L,M,R,nu,sizey,ny);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  SIMULATION
%%% initialise data
nx = size(A,1);nuu=size(B,2);
x=[x0,x0];xm=x;xp=x;
yp=C*xm;ym=yp;
u=zeros(sizey,2);r=yp*0;

runtime = size(ref,2)-1;

for i=2:runtime;
    
   r(:,i+1) = ref(:,i+1);

   %%%%% Update offset 
   offset(:,i) = yp(:,i) + noise(:,i)-ym(:,i);   %%% Noise effects measurement only
   
   %%%% Update control decisions using a quadratic program
   ufut = -K*xm(:,i) +Pr*(r(:,i+1)-offset(:,i));   %%  (unconstrained)
   u(:,i) = ufut(1:sizey);
   Du(:,i)=u(:,i)-u(:,i-1);

   %%%% Simulate model
      xm(:,i+1) = A*xm(:,i)+B*u(:,i);
      ym(:,i+1) = C*xm(:,i+1);
   %%%% Simulate process
      xp(:,i+1) = A*xp(:,i)+B*u(:,i);
      yp(:,i+1) = C*xp(:,i+1)+dist(:,i);
end

%%%%% Ensure data lengths are all compatible
u(:,i+1)=u(:,i);
Du(:,i+1)=Du(:,i)*0;
r(:,i+1) = ref(:,i+1);
noise = noise(:,1:i+1);
d = dist(:,1:i+1);

x=xp;y=yp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%  Produce a neat plot
time=0:size(u,2)-1;
for i=1:size(B,2);
    figure(i);clf reset
    plotall(yp(i,:),r(i,:),u(i,:),Du(i,:),d(i,:),noise(i,:),time,i);
end

disp('**************************************************');
disp(['***    There are ',num2str(size(B,2)),' figures    ***']);
disp('**************************************************');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Function to do plotting in the MIMO case and 
%%%%% allow a small boundary around each plot

function plotall(y,r,u,Du,d,noise,time,loop)

subplot(221);plot(time,y','-',time,r','--');
xlabel(['IMGPC - Outputs and set-point in loop ',num2str(loop)]);
subplot(222);plot(time,Du','-');
xlabel(['IMGPC - Input increments in loop ',num2str(loop)]);
subplot(223);plot(time,u','-');
xlabel(['IMGPC - Inputs in loop ',num2str(loop)]);
subplot(224);plot(time,d','b',time,noise,'g');
xlabel(['IMGPC - Disturbance/noise in loop ',num2str(loop)]);

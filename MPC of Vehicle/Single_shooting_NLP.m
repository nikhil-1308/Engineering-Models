% point stabilization + Single shooting
clear all
close all
clc

% CasADi v3.4.5
addpath('C:\Users\NIKHIL\Desktop\casadi-windows-matlabR2016a-v3.5.5')
import casadi.*

T = 0.2; % sampling time [s]
N = 25; % prediction horizon
rob_diam = 0.3;

v_max = 0.6; v_min = -v_max;
omega_max = pi/4; omega_min = -omega_max;

x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta');
states = [x;y;theta]; n_states = length(states);

v = SX.sym('v'); omega = SX.sym('omega');
controls = [v;omega]; n_controls = length(controls);
rhs = [v*cos(theta);v*sin(theta);omega]; % system r.h.s

f = Function('f',{states,controls},{rhs}); % nonlinear mapping function f(x,u)
U = SX.sym('U',n_controls,N); % Decision variables (controls)
P = SX.sym('P',n_states + n_states); 
% first half is initisl states & second half is eference states 
% parameters (which include the initial and the reference state of the robot)
% P=[P_0,P_1,P_2,P_3,P_4,P_5]

X=SX.sym('X',n_states,(N+1));

% compute solution symbolically
X(:,1)=P(1:3); % fill the matrix X with the elements in P, column wise
for k=1:N
    st=X(:,k); con=U(:,k);
    f_value=f(st,con);
    st_next=st+(T*f_value);
    X(:,k+1)=st_next;
end
%  X=[[X_0, X_3, X_6, X_9],
%     [X_1, X_4,X_7,X_10],
%     [X_2,X_5,X_8,X_11]

% this function to get the optimal trajectory knowing the optimal solution
ff=Function('ff',{U,P},{X});

obj = 0; % Objective function
g = [];  % constraints vector

Q = zeros(3,3); Q(1,1) = 1;Q(2,2) = 5;Q(3,3) = 0.1; % weighing matrices (states)
R = zeros(2,2); R(1,1) = 0.5; R(2,2) = 0.05; % weighing matrices (controls)
% compute objective
for k=1:N
    st = X(:,k);  con = U(:,k);
    obj = obj+(st-P(4:6))'*Q*(st-P(4:6)) + con'*R*con; % calculate obj
end

% compute constraints
for k = 1:N+1   % box constraints due to the map margins
    g = [g ; X(1,k)];   %state x
    g = [g ; X(2,k)];   %state y
end

% make the decision variables one column vector
% U=[[U_0,U_2,U_4],
%     [U_1,U_3,U_5]]
% OPT_variables =[U_0, U_1,  U_2, U_3,  U_4, U_5]
%                  v   omega  v   omega  v    v
OPT_variables = reshape(U,2*N,1);
nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 100;
opts.ipopt.print_level =0;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

args=struct;args = struct;
% inequality constraints (state constraints)
args.lbg = -2;  % lower bound of the states x and y
args.ubg = 2;   % upper bound of the states x and y 

% input constraints
args.lbx(1:2:2*N-1,1) = v_min; % every odd element is v
args.lbx(2:2:2*N,1)   = omega_min; % every even element is omega
args.ubx(1:2:2*N-1,1) = v_max; % every odd element is v
args.ubx(2:2:2*N,1)   = omega_max;

%----------------------------------------------
% ALL OF THE ABOVE IS JUST A PROBLEM SETTING UP


% THE SIMULATION LOOP SHOULD START FROM HERE
%-------------------------------------------

t0=0;
x0=[0;0;0.0]; % initial condition
xs=[1.5;1.5;0.0]; % reference posture.
xx(:,1)=x0; % XX contains the history of states
t(1)=t0;
u0=zeros(N,2); % two control inputs
sim_tim=20; % maximum simulation time

% start MPC
mpciter=0;
xx1=[];
u_cl=[];

% the main simulaton loop... it works as long as the error is greater
% than 10^-2 and the number of mpc steps is less than its maximum
% value.
main_loop = tic;
while(norm((x0-xs),2)>1e-2 && mpciter<sim_tim/T)
    args.p = [x0;xs]; % set the values of the parameters vector
    % reshaping to form a vector
    args.x0 = reshape(u0',2*N,1); % initial value of the optimization variables
    %tic
    
    % minimize cost function "J" w.r.t 'u'
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
            'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u=reshape(full(sol.x)',2,N)'; % reshaping it as a matrix
    % get solution TRAJECTORY
    ff_value=ff(u',args.p);
    xx1(:,1:3,mpciter+1)=full(ff_value)'; % minimized cost funtion output x(k+1)=f(xu(k),u(k))
    
    u_cl=[u_cl;u(1,:)]; % storing the control action, always the control action will be the first term
    % feeding the control action to the system
    t(mpciter+1)=t0;
    [t0,x0,u0]=shift(T,t0,x0,u,f); % the u0 from here will be used again in arg.x0
    xx(:,mpciter+2)=x0;
    mpciter
    mpciter=mpciter+1;
end

main_loop_time = toc(main_loop);
ss_error = norm((x0-xs),2)
average_mpc_time = main_loop_time/(mpciter+1)

Draw_MPC_point_stabilization_v2 (t,xx,xx1,u_cl,xs,N,rob_diam) % a drawing function
clear all;
close all;
clc;

% objective=@(x) x(1)*x(4)*(x(1)+x(2)+x(3));
% objective=@(x) sin(x(1))+0.1*x(2)^2+0.05*x(1)^2;
% objective=@(x)[x(2);x(3);0.05*x(1)*(x(2)+x(3))];
% x0=[1,5,5,1];
x0=[-3;-3];
% disp(['Initial objective: ' num2str(objective(x0))]);
A=[];
b=[];
Aeq=[];
beq=[];
% lb=1*ones(4);
lb=[-5;-3];
% ub=5*ones(4);
ub=[1;3];
nonlincon=@nlcon;
% 
% [x,fval,exfl,output,lambda]=fmincon(objective,x0,A,b,Aeq,beq,lb,ub,nonlincon);
% 
% disp(x)
% 
% disp(['Final objective: ' num2str(objective(x))]);
% [c,ceq]=nlcon(x)

options=optimoptions('fmincon','Algorithm','sqp','Display','iter-detailed',...
    'MaxFunctionEvaluations',100000,'MaxIterations',2000,...
    'FunctionTolerance',1e-10);

[X,cost]=fmincon(objective,x0,A,b,Aeq,beq,lb,ub,nonlincon,options)

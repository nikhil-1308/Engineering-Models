clear all;
close all;
clc;

N=100;
f=@(x)[x(1)-0.5*sqrt(x(1));x(2)+0.2*sqrt(x(1))-0.3*sqrt(x(2))];
h=@(x)[x(2)];
B=[0.4;0];
x=[0;0];
u=[zeros(1,10),ones(1,N-10)];

for k=1:N
    x(:,k+1)=f(x(:,k))+B*u(:,k);
end

for k=1:100
    X(:,k) = fmincon(@Output, [x(1,k) x(2,k)], [], [], [], [], [0 0], [1 1],@mycons);
end

% [X,FVAL,EXITFLAG,OUTPUT,LAMBDA]=fmincon(@Output,[0 0],[],[],[],[],[0 0],[1 1],@mycons);


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

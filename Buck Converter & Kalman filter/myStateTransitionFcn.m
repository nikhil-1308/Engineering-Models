function x = myStateTransitionFcn(x,u)
dt=0.05;

R=20;
L=3.23;
c=2.3;


% A=[0 -(1/L);-(1/c) 1/(R*c)];
% B=[1/L;0];

x=[-(1/L)*x(2)+(1/L)*u; -(1/C)*x(1)+(1/(R*c)*x(2))]*dt;
end
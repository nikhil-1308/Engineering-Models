function y = myMeasurementFcn(x,u)
R=20;
L=3.23;
c=2.3;

% dt=1;
% A=[0 -(1/L);-(1/c) 1/(R*c)];
% B=[1/L;0];
y=x(2,1)*u;
% x=[-(1/L)*x(1)+(1/L)*u; -(1/c)*x(1)+(1/(R*c)*x(1))]*dt;

end
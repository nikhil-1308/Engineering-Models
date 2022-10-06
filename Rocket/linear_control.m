close all;
clear all;
clc;

global K
theta = 30;
theta_dot = 0;
Z = 0;
alpha = 40;
alpha_w = 5;
delta = 0;
Q0 = [theta;theta_dot;Z;alpha;alpha_w;delta];
T = 0:0.1:1000; % solution time mesh

m = 38901;
Tc = 2.361e+6;
T0 = 0.1*Tc;
V = 1347;
Vm = 30;
roh = 0.0765;
Dyp = 0.5*roh*Vm^2;
Ca = 1.7228;
Fbase = 10;
S = 116.2;
D = Ca*Dyp*S - Fbase;
F = T0 + Tc;
Malpha = 0.3807;
Mdelta = 0.526;
Nalpha = 686819;
A = [0 1 0;...
     Malpha 0 Malpha/V;...
     -(F-D+Nalpha)/m 0 -Nalpha/(m*V)];
B = [0;Mdelta;Tc/m];
C = [1 1 0];
Qq = C'*C;
R = 0.2^2;
[K, S, E] = lqr(A, B, Qq, R);
%%
[t,y] = ode45(@dxdt, T, Q0);
plot(t,y(:,1))

function Q = dxdt(~,Q)
global K
m = 38901;
Tc = 2.361e+6;
T0 = 0.1*Tc;
V = 1347;
Vm = 30;
roh = 0.0765;
Dyp = 0.5*roh*Vm^2;
Ca = 1.7228;
Fbase = 10;
S = 116.2;
D = Ca*Dyp*S - Fbase;
F = T0 + Tc;
Malpha = 0.3807;
Mdelta = 0.526;
Nalpha = 686819;
% K1 = 0.5441;
% K2 = 5.8607;
% K3 = 0.5464;
K1 = K(1);
K2 = K(2);
K3 = K(3);
B0 = ((Tc*K1)/(m*V))*(Malpha+(Mdelta*Nalpha)/Tc) - ((F-D)/(m*V))*(Mdelta*K3-Malpha);
B1 = Mdelta*(K1+K2) - Malpha + ((K2*Tc)/(m*V))*(Malpha+(Mdelta*Nalpha)/Tc);
B2 = Mdelta*K2 + (Tc/(m*V))*(K3+(Nalpha/Tc));
alpha = Q(1,:) + Q(3,:)/V + Q(5,:);
Q(6,:) = -K1*Q(1,:) - K2*Q(2,:) - K3*alpha;
% theta theta_dot Z
Q(1:3,:) =  [0 1 0;...
             Malpha 0 Malpha/V;...
             -(F-D+Nalpha)/m 0 -Nalpha/(m*V)]*Q(1:3,:) + [0;Mdelta;Tc/m]*Q(6,:) + [0;Malpha;-Nalpha/m]*Q(5,:);
% alpha alpha_w
Q(4:5,:) = [-B0/B1 (Malpha*K1)/B1;...
            B1/(Malpha*K2) -K1/K2]*Q(4:5,:);
Q(4:5,:) = [-B1/B2 (Malpha*K1)/B2;...
            B1 -Mdelta*K1]*Q(4:5);
Q(4,:) = -B2*Q(4,:);

end
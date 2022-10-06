close all;
clear all;
clc;

qr = [0;0];
q = [0.7769; 0.0598; 0; 0.6268,];
Q0 = [qr;q];
T = 0:1:1000; % solution time mesh

[t,y] = ode45(@dxdt, T, Q0);

function Q = dxdt(~,Q)
a = 0.9958;
p = 0.005;
Kp = 1.5441;
Kd = 0.8607;
Q(1:6,:) =  [a*Q(2,:)*p-Kp*Q(4,:)-Kd*Q(1,:);...
             -a*p*Q(1,:)-Kp*Q(5,:)-Kd*Q(2,:);...
             (Q(2,:)/2)*Q(4,:)-(Q(1,:)/2)*Q(5,:)+(p/2)*Q(6,:);...
             -(Q(2,:)/2)*Q(3,:)+(p/2)*Q(4,:)+(Q(1,:)/2)*Q(6,:);...
             (Q(1,:)/2)*Q(3,:)-(p/2)*Q(4,:)+(Q(2,:)/2)*Q(6,:);...
             -(p/2)*Q(3,:)-(Q(2,:)/2)*Q(4,:)-(Q(2,:)/2)*Q(5,:)];

end
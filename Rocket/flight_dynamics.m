close all;
clear all;
clc;

qc = [0.1;0;0;0];
q = [0.3594;-0.6089;-0.3625;0.6072];
pqr = [3.4916e-5;6.4018e-5;0;0];
qe = [0;0;0;0];
eu_ang = [0;0;0];
xyz = [-8.7899e+4;-1.8385e+7;9.9605e+6];
uvw = [1340.65;-6.41;0];
Q0 = [qc;q;pqr;qe;eu_ang;xyz;uvw];
T = 0:100; % solution time mesh

[t,y] = ode45(@dxdt, T, Q0);

function Q = dxdt(~,Q)
% qe
Q(13,:) = Q(4,:)*Q(5,:) + Q(3,:)*Q(6,:) - Q(2,:)*Q(7,:) - Q(1,:)*Q(8,:);
Q(14,:) = -Q(3,:)*Q(5,:) + Q(4,:)*Q(6,:) + Q(1,:)*Q(7,:) - Q(2,:)*Q(8,:);
Q(15,:) = Q(3,:)*Q(5,:) - Q(1,:)*Q(6,:) + Q(4,:)*Q(7,:) - Q(3,:)*Q(8,:);
Q(16,:) = Q(1,:)*Q(5,:) + Q(2,:)*Q(6,:) + Q(3,:)*Q(7,:) + Q(4,:)*Q(8,:);
% q
Q(5:8,:) = [Q(12,:) Q(11,:) -Q(10,:) Q(9,:);...% pqr
            -Q(10,:) Q(12,:) Q(9,:) Q(10,:);...% pqr
            Q(10,:) -Q(9,:) Q(12,:) Q(11,:);...% pqr
            -Q(9,:) -Q(10,:) Q(11,:) Q(12,:)]*Q(5:8,:);
I = [1.2634e+6 -1.5925e+3 5.5250e+4;...
     -1.5925e+3 2.8797e+8 -1.5263e+3;...
     5.5250e+4 -1.5263e+3 2.8798e+8];
c = [-220.31 0.02 0.01];
Xa = -275.6;
S = 116.2;
roh = 0.0765;
% eu_angle
Q(17:19,:) = (1/cosd(Q(18,:)))*[cosd(Q(18,:)) sind(Q(17,:))*sind(Q(18,:)) cosd(Q(18,:))*sind(Q(17,:));...
                                0 cosd(Q(18,:))*cosd(Q(17,:)) -sind(Q(18,:))*cosd(Q(17,:));...
                                0 sind(Q(18,:)) cosd(Q(18,:))]*Q(9:11,:);
q1 = sind(Q(17,:)/2)*cosd(Q(18,:)/2)*cosd(Q(19,:)/2) - cosd(Q(17,:)/2)*sind(Q(18,:)/2)*sind(Q(19,:)/2);
q2 = cosd(Q(17,:)/2)*sind(Q(18,:)/2)*cosd(Q(19,:)/2) + sind(Q(17,:)/2)*cosd(Q(18,:)/2)*sind(Q(19,:)/2);
q3 = cosd(Q(17,:)/2)*cosd(Q(18,:)/2)*sind(Q(19,:)/2) - sind(Q(17,:)/2)*sind(Q(18,:)/2)*cosd(Q(19,:)/2);
q4 = cosd(Q(17,:)/2)*cosd(Q(18,:)/2)*cosd(Q(19,:)/2) + sind(Q(17,:)/2)*sind(Q(18,:)/2)*sind(Q(19,:)/2);

Cbi = [1-2*(q2^2+q3^2) 2*(q1*q2+q3*q4) 2*(q1*q3-q2*q4);...
       2*(q1*q2-q3*q4) 1-2*(q1^2+q3^2) 2*(q2*q3+q1*q4);...
       2*(q1*q3+q2*q4) 2*(q2*q3-q1*q4) 1-2*(q1^2+q2^2)];
we = 7.2921e-5;
vw = 30;
Vw = [vw*deg2rad(Q(17,:));vw*deg2rad(Q(18,:));vw*deg2rad(Q(19,:))];
Vm = Cbi*[0 we 0;...
          we 0 0;...% xyz
          0 0 0]*Q(20:22,:) - Vw;
alpha = tand(Vm(3,:)/Vm(1,:));
Vm_mag = norm(Vm');
beta = tand(Vm(2,:)/Vm_mag);
Dyp = 0.5*roh*Vm_mag^2;
Ca = 1.7228;
Cybeta = -0.1752;
Cn0 = 0.3807;
Cnalpha = 0.1465;
Fbase = 10;
D = Ca*Dyp*S - Fbase;
C = Cybeta*beta*Dyp*S;
N = (Cn0 + Cnalpha*alpha)*Dyp*S;
b = 12.16;
Cmp0 = 0.0556;
Cmpalpha = 0.0521;
Cmrbeta = 0.0423;
Cmybeta = 0.0416;
Faero = [-D;C;-N];
Taero = [0 c(3) -c(2);...
         -c(3) 0 -Xa+c(1);...
         c(2) Xa-c(3) 0]*Faero + [Cmrbeta*Dyp*S*b;(Cmp0 + Cmpalpha)*Dyp*S*b;Cmybeta*Dyp*S*b];
T = 2.3608e+6;
Kpy = 1.3484;
Kdy = 1.5023;
Kpz = 1.3484;
Kdz = 1.5023;
Kpx = 1.5441;
Kdx = 0.8607;
Kiy = 0.3416;
Trcs = -Kpx*2*Q(13,:) - Kdx*Q(9,:);
delta_y = Kpy*2*Q(14,:) - Kiy*Q(14,:) - Kdy*Q(10,:);
% delta_y = -2*Kpy*2*Q(14,:) - Kdy*Q(10,:);
delta_z = -Kpz*2*Q(15,:) - Kdz*Q(11,:);
Frkt = [T;-T*delta_z;-T*delta_y];
Xg = -296;
Trkt = [0 c(3) -c(2);...
        -c(2) 0 -Xg+c(1);...
        c(2) Xg-c(3) 0]*Frkt;
m = 38901;
mu = 1.407644176e+16;
r_cap = Q(20:22,:);
r = norm(r_cap');
Re = 2.0925646e+7;
J2 = 1.082631e-3;
J3 = -1.255e-6;
J4 = -1.61e-6;
C1 = -1 + (Re^2/r^2)*(3*J2*((3/2)*sin(Q(18,:))^2 - 0.5) + 4*J3*(Re/r)*((5/2)*sin(Q(18,:))^3 -...
                        (3/2)*sin(Q(18,:))) + 5*J4*(Re/r^2)*((35/8)*sin(Q(18,:))^4 + (3/8)));
C2 = J2*3*sin(Q(18,:)) + (Re/r)*J3*((15/2)*sin(Q(18,:))^2 - (3/2)) + (Re^2/r)*J4*((35/2)*sin(Q(18,:))^3 - (15/2)*sin(Q(18,:)));
g = (mu/r^2)*(C1*[Q(20,:)/r;Q(21,:)/r;Q(22,:)/r] - C2*[0 -Q(22,:)/r Q(21,:)/r;...
                                                       Q(22,:)/r 0 -Q(20,:)/r;...
                                                       -Q(21,:)/r Q(20,:)/r 0]*[-Q(21,:)/r;Q(20,:)/r;0]);
Ftotal_b = Faero + Frkt + [Trcs/b;0;0];
Ftotal_i = Cbi'*Ftotal_b;
Q(20:22,:) = (1/m)*Ftotal_i + g; % xyz
% pqr
Q(9:11,:) = inv(I)*([Q(12,:) -Q(11,:) Q(10,:);...
                     Q(11,:) Q(12,:) -Q(9,:);...
                     -Q(10,:) Q(9,:) Q(12,:)]*I*Q(9:11,:) + Taero + Trkt + [Trcs;0;0]);
Q(23:25,:) = [Q(12,:) -Q(11,:) Q(10,:);...
              Q(11,:) Q(12,:) -Q(9,:);...
              -Q(10,:) Q(9,:) Q(12,:)]*Q(23:25,:) + Cbi*g + (1/m)*Faero + (1/m)*Frkt;

end


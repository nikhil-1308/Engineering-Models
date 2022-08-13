%run this model first

%motor
L=0.003;
R=0.141;
Kb=0.00574;
Km=Kb;
J=0.0001;
b=3.97e-6;
fc=0.0;

%car/traction parameters
m=1.136;
Rw=3.2e-2;
g=9.81;
Q=0.37;
GR=2.5;

%initial conditions
i_0=0;
wm_0=0;
tht_0=0;
v_0=0;
xp_0=0;

%tolerance
tol=1.0e-10;

%desired speed(speed up to)
wm_des=275;

%desired speed(speed down to)
wm_des2=110;

%input voltage
u1=12; %12v(constant)

out.FT=1200*9.81*0.85;
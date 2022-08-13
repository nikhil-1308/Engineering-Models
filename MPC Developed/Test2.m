clear all;
close all;
clc;

% A =[0.8      0.4 1;
%     -0.2    0.7 -0.2;1.2 0.3 0.2];
% B =[0.4;0.8;0];
% C =[1.9 2 -0.2];

% %% CAR MODEL
% V=14;
% m_s=240;
% m_u=36;
% c_s=980;
% k_s=16000/100;
% k_t=160000/100 /,/;
% Iw=2;
% T=20;
% R_r=30;
% K_k=1;
% a=1;
% % linear state space representation
% 
% A=[     0           1              0                    0                 0        ;
%     -k_s/m_s    -c_s/m_s        k_s/m_s              c_s/m_s              0        ;
%         0           0              0                    1                 0        ;
%     k_s/m_u     c_s/m_u         (-k_s-k_t)/m_u      -c_s/m_u              0        ;
%        0           0           (R_r*K_k*a)/(V*Iw)       0    -(R_r^2*K_k*a)/(V*Iw)  ];
%    
% B=[0; 0; 0; k_t/m_u; -(R_r*K_k*a)/(V*Iw)];
% 
% C=[0 0 0 0 1];

% FLIGHT MODEL

% A= [-0.313 56.7 0;
%     -0.0139 -0.426 0;
%     0 56.7 0];
% 
% B=[0.232; 0.0203; 0];
% 
% C=[0 0 1];
% 
% D=0;

A =[0.9146         0    0.0405;
    0.1665    0.1353    0.0058;
         0         0    0.1353];
B =[0.0544   -0.0757;
    0.0053    0.1477;
    0.8647         0];
C =[1.7993   13.2160         0;
    0.8233         0         0];
D=zeros(2,2);

Q=C'*C;Q2=Q;
R=2*eye(1); R2=R*10;
nx=3;
nu=1;
K0=dlqr(A,B,Q2,R2);
ref=[zeros(1,10),ones(1,50)];
dist=[zeros(1,30),-0.5*ones(1,30)];
noise = [zeros(1,10),randn(1,90)*0.13+1]; %%% noise

%%%%% constraints
umin=[-1;-2] ;    %%% umin<u<umax
umax=[2;1.5];
Kxmax=[1 0.2 0;-0.1 0.4 0;-1,-0.2 0;0.1,-0.4 0;0 0 1;0 0 -1];
xmax=[4;4;2.5;0.5;2;2];


%%% Horizon 1
nc=2;runtime=49;x0=[0;0;0];
[J,x,y,u,c,KSOMPC] = chap5_ompc_simulate_constraints_b(A,B,C,D,nc,Q,R,Q2,R2,x0,runtime,ref,dist,noise,umin,umax,Kxmax,xmax);
% [J,x,y,u,c,KSOMPC] = chap4_ompc_simulateb(A,B,C,D,nc,Q,R,Q2,R2,x0,runtime,ref,dist,noise);
figure(1); clf reset
v=2:length(y);
% v1=2:length(ym);


% subplot(211);
plot(v,y(v),'b','linewidth',2); title(['SOMPC output for n_c=',num2str(nc)],'fontsize',18)
hold on; plot(v,ref(v),'--r','linewidth',2);
plot(v,noise(v),'--k','linewidth',2);
plot(v,dist(v),'g','linewidth',2);
plot(v,u(v),'m','linewidth',2);
legend('Output','target','noise','disturbance','input');
% subplot(212);plot(v,J(v),'b','linewidth',2);title('cost is monotonic','fontsize',18)
% subplot(313);
% c=[c;zeros(20,size(c,2))];
% for k=9:nc+12;
%     alp=(nc+12-k)/(3+nc);
%     plot(k+1:21+nc,c(1:20+nc-k+1,k)','linewidth',2,'color',[alp,0,1-alp]);hold on
%     title('Optimised cfut for SOMPC','fontsize',18)
% end
% legend('c(k)','c(k+1)','c(k+2)','c(k+4)','c(k+5)')

error=norm(ref(v)-y(v))

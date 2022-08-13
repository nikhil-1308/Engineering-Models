clear all;
close all;
clc;

N=100;
f=@(x)[x(1);x(1)*x(2)];
h=@(x)[x(2)];
x=[0.9416+0.3059*j; 1.1895+0.8642*j];
xp(:,1)=x(:,1);
xu(:,1)=x(:,1);
P=eye(2);
q=0.001;
Q=q*q*eye(2); % Porocess nois
r=0.2;
R=r*r; % Measurement nois


nx=4;

alpha=0.001;
beta=2;
keppa=0;

lamb=alpha^2*(nx+keppa)-nx;

wm(:,1)=lamb/(lamb+nx);
wc(:,1)=wm+(1-alpha^2+beta);

for k=1:N
    wm(:,1:k+1)=1/2*(nx+lamb);
    wc(:,1:k+1)=1/2*(nx+lamb);
    gamma(:,k+1)=sqrt(nx+lamb);

end

wc=wc;
wm=wm;

for k=1:N
    x(:,k+1)=f(x(:,k));
    y(k+1)=h(x(:,k+1));
end

%% ***** START *****
for k=2:N
for k=2:N
    xhat(k+1)=wm(k)*x(k+1);
end

x_hat=sum(xhat);


for k=2:N
    p(k)=wc(k)*(x(k+1)-x_hat)*(x(k+1)-x_hat)';%Q
end

P_cov=sum(p);

sigma1=[];
sigma2=[];

for k=2:N
    sigma1(k)=x_hat+sqrt(nx+lamb)*P_cov;
    sigma2(k)=x_hat-sqrt(nx+lamb)*P_cov;
end

% sigma=reshape([sigma1;sigma2], size(sigma1,1), []);

sigma_1=sum(sigma1);
sigma_2=sum(sigma2);


Xt=[sigma_1;sigma_2];

for k=1:N+1
   Xt(:,k+1)=f( Xt(:,k));
   
end


for k=2:N+1
    X_hat(:,k+1)=wm(:,k)*Xt(:,k+1);
end

x_hat=sum(x_hat);

for k=2:N+1
   Yt(k+1)=h(Xt(:,k+1));
end

for k=2:N+1
    y_hat(k+1)=wm(k)*Yt(k+1);
end

y_hat=sum(y_hat);


for k=2:N+1
    Pyy(k)=wc(:,k)*(Yt(k)-y_hat)*(Yt(k)-y_hat)';%R
end

P_yy=sum(Pyy);


for k=2:N+1
    Pxy(k)=wc(:,k)*(Xt(k)-x_hat)*(Yt(k)-y_hat)';
    
end

P_xy=sum(Pxy);

for k=2:N+1
    Xt(:,k)=x_hat+P_xy*P_yy'*(Yt(:,k)-y_hat);
    p(k)=P_cov-P_xy*P_yy'*P_xy';
end


end



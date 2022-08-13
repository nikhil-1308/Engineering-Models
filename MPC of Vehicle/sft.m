function [t0, x0, u0] = sft(T, t0, x0, U_sol,f)
% add noise to the control actions before applying it
con_cov = diag([0.005 deg2rad(2)]).^2;
con = U_sol(1,:)' + sqrt(con_cov)*randn(2,1); 
st = x0;

f_value = f(st,con);   
st = st+ (T*f_value);

x0 = full(st);
t0 = t0 + T;

u0 = [U_sol(2:size(U_sol,1),:);U_sol(size(U_sol,1),:)]; % shift the control action 
end
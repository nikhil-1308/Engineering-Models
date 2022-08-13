function [c,ceq]=nlcon(x)
% % c=25-x(1)*x(2)*x(3)*x(4);
% % ceq=sum(x.^2)-40;
c=[];
ceq=(x(1)+3)^3-x(2);
% c=@(x)(x(2)+0.2*sqrt(x(1))-0.3*sqrt(x(2)))-1;
% ceq=[];
end


%% timer ARDtimer.m
cmp=cmp+1;
y(cmp)=a.readVoltage('A4');
% a.writePWMVoltage('D3',y(cmp));
realTime(cmp,:)=clock;
% x(cmp)=cmp-1;
% h=plot(x,y);
% grid on
% h.LineWidth=2;
% axis([0 300 2 3]);
% drawnow;
N=[1 4 8 16 32]; %��Ⱥ����
M=[10 50 100 200];
yb=combine(N);
boxplot(yb','Whisker',100000,'symbol',' ');
%  set(gca,'xticklabels',{ str11});%id=1
% 10 100 200 300%

%ylim([-2450,-1350]);
set(gca,'xticklabels',{ 'N=1','M=5', '10', '15', '20','M=5', '10', '15', '20','M=5', '10', '15', '20','M=5', '10', '15', '20','200'});%id=3

set(gca,'TickLabelInterpreter','latex');
set(gca, 'XTickLabelRotation', 45);% ��������б45��
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);% ������������
xlabel('~\emph{N}=4~~~~~~~~~~~~~\emph{N}=8~~~~~~~~~~~~~\emph{N}=16~~~~~~~~~~~~~\emph{N}=32~~~~~~~','Interpreter','latex') 
ylabel('$f_{\rho}(\underline{y})$','FontSize',15,'Interpreter','latex')
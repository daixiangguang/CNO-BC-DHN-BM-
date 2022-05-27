N=[1 4 8 16 32]; %��Ⱥ����
M=[10 50 100 200]; %���������
yb=combine(N);
boxplot(yb','symbol',' ');% N=[1 4 8 16 32]; %��Ⱥ����
%  set(gca,'xticklabels',{ str11});%id=1
% 10 100 200 300%

ylim([-19700,-19100]);
set(gca,'xticklabels',{ 'N=1','M=10', '50', '100', '200','M=10', '50', '100', '200','M=10', '50', '100', '200','M=10', '50', '100', '200','400'});%id=3

set(gca,'TickLabelInterpreter','latex');
set(gca, 'XTickLabelRotation', 45);% ��������б45��
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);% ������������
xlabel('~\emph{N}=4~~~~~~~~~~~~~\emph{N}=8~~~~~~~~~~~~~\emph{N}=16~~~~~~~~~~~~~\emph{N}=32~~~~~~~','Interpreter','latex') 
ylabel('$f_{\rho}(\underline{y})$','FontSize',15,'Interpreter','latex')
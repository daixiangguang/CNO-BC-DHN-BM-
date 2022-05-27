N=[1 16 32 64 96]; %��Ⱥ����
M=[10 100 200 300]; %���������
yb=combine(N);
boxplot(yb','symbol',' ');% N=[1 4 8 16 32]; %��Ⱥ����
%  set(gca,'xticklabels',{ str11});%id=1
% 10 100 200 300%

ylim([-108550,-103000]);
set(gca,'xticklabels',{ 'N=1','M=10', '100', '200', '300','M=10', '100', '200', '300','M=10',  '100', '200', '300','M=10', '100', '200', '300','500'});%id=3

set(gca,'TickLabelInterpreter','latex');
set(gca, 'XTickLabelRotation', 45);% ��������б45��
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);% ������������
xlabel('~\emph{N}=16~~~~~~~~~~~~~\emph{N}=32~~~~~~~~~~~~~\emph{N}=64~~~~~~~~~~~~~\emph{N}=96~~~~~~~','Interpreter','latex') 
ylabel('$f_{\rho}(\underline{y})$','FontSize',15,'Interpreter','latex')
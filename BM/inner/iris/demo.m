clc
clear
close all
load('iris.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,2);
[n,m]=size(fea);
options.ReducedDim=fix(0.9*m);
W = PCA(fea,options);
[n,~]=size(fea);
fea=fea*W;
% init d
d=gaussinKernel(fea,0.08);
d=-d;
d=d-diag(diag(d));

%BM_PSO_Hopfiled_Parameter 
options1.cores=1;
options1.pop=1;
options1.alpha=5;
options1.beta=5;
options1.iterations=20000;
options1.p=nClass;
options1.C1=2;
options1.C2=2;

[s,obj]=BM_PSO_Hopfield(d,options1);
l=label1(s,n);
plot(obj(1:100,:),'k');
hold on,
xlabel('Inner-loop Iteration','fontsize',15);
ylabel('$f_{\rho}(\underline{y})$','fontsize',15,'Interpreter','latex'); 








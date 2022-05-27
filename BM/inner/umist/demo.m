clc
clear
close all
load('umist.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,1);
[n,m]=size(fea);
options.ReducedDim=fix(0.9*m);
W = PCA(fea,options);
[n,~]=size(fea);
fea=fea*W;
% init d
d=gaussinKernel(fea,0.2);
d=-d;
d=d-diag(diag(d));

%BM_PSO_Hopfiled_Parameter 
options1.cores=1;
options1.pop=1;
options1.alpha=15;
options1.beta=15;
options1.iterations=20000;
options1.p=nClass;
options1.C1=2;
options1.C2=2;

[s,obj]=BM_PSO_Hopfield(d,options1);
l=label1(s,n);
index=randperm(2000);
index=sort(index);
index=index';
obj=[obj,index];

plot(obj(1:200,1),'k');
xlabel('Inner-loop Iteration','fontsize',15);
ylabel('$f_{\rho}(\underline{y})$','fontsize',15,'Interpreter','latex'); 
axes('position',[0.5 0.4 0.37 0.37]);
plot(obj(100:200,2),obj(100:200,1),'k');








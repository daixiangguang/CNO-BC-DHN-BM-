
 load('umist.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,1);
%parpool(64);
% fea=mapminmax(fea,0,1);
%fea=zscore(fea);
[n,m]=size(fea);
options.ReducedDim=fix(0.9*m);
W = PCA(fea,options);
[n,~]=size(fea);
fea=fea*W;
lambda=10;
beta=0.1;
gamma=0.1;
iter=100;
k=20;
% QAP
   d=gaussinKernel(fea,0.2);
  d=-d;
   d=d-diag(diag(d));
QAP=zeros(k,3);
% y=cell(k,1);
pop=32;
p=max(label);
initialx=zeros(n*p,pop);
for i=1:pop
    temp=randn(n*p,1);
    temp(temp>=0)=1;
    temp(temp<0)=-1;
    initialx(:,i)=temp;
end
initialx=initialx';
for i=1:1
    i
    
    [y,x]=kernel(initialx);
     y=y(y~=0);

end
ITER=10;
yb=y_processing(y,ITER);
index=randperm(3600);
index=sort(index);
index=index';
plot(0:15,yb,'k');
hold on,
xlabel('Inner-loop Iteration','fontsize',15);
ylabel('$f_{\rho}(\underline{y})$','fontsize',15,'Interpreter','latex'); 
axes('position',[0.5 0.4 0.37 0.37]);
plot(index(5:15,:),yb(5:15,1),'k');
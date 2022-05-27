
load('iris.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,1);
%parpool(64);
% fea=mapminmax(fea,0,1);
% fea=zscore(fea);
[n,m]=size(fea);
options.ReducedDim=fix(0.9*m);
W = PCA(fea);
[n,~]=size(fea);
fea=fea*W;
lambda=10;
beta=0.1;
gamma=0.1;
iter=100;
k=10;
% QAP
d=gaussinKernel(fea,0.05);
d=-d;
d=d-diag(diag(d));
QAP=zeros(k,3);
% d=squareform(pdist(fea));
% fea=d;
temp=0;
tic;
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



[y,x]=kernel(initialx);

yb=dataprocessing(y);
index=randperm(1000);
index=sort(index);
index=index';
yb=yb(:,1:51);
[row,column]=size(yb);
for i=1:size(yb,1)
    if i~=row
        plot(0:50,yb(i,:));
        hold on;
    else
        plot(0:50,yb(i,:),'r');
        hold on;
    end
    xlabel('Outer-loop Iteration','fontsize',15);
    ylabel('$f_{\rho}(\underline{y})$','fontsize',15,'Interpreter','latex');   
end
axes('position',[0.62 0.4 0.25 0.27]);
plot(index(1:10,:),yb(end,1:10),'r');



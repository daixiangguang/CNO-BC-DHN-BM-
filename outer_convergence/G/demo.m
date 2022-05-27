
load('G.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,1);
%parpool(64);
% fea=mapminmax(fea,0,1);
 fea=zscore(fea);
[n,m]=size(fea);
options.ReducedDim=fix(0.9*m);
W = PCA(fea,options);
[n,~]=size(fea);
fea=fea*W;
lambda=10;
beta=0.1;
gamma=0.1;
iter=100;
k=10;
% QAP
d=gaussinKernel(fea,4);
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


index=randperm(9300);
index=sort(index);
index=index';
[y,x]=kernel(initialx,d(:));
yb=dataprocessing(y);
%[row,column]=size(yb);
ya=yb(:,1:9001);
[row,column]=size(ya);
for i=1:size(ya,1)
    if i~=row
        plot(0:9000,ya(i,:));
        hold on;
    else
        plot(0:9000,ya(i,:),'r');
        hold on;
    end
    xlabel('Outer-loop Iteration','fontsize',15);
     ylabel('$f_{\rho}(\underline{y})$','fontsize',15,'Interpreter','latex');
    set(gca,'ycolor','k');
     set(gca,'xcolor','k');
end
axes('position',[0.62 0.4 0.25 0.27]);
plot(index(6900:8000,:),yb(end,6900:8000),'r');



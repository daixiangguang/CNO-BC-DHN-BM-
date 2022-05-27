clc
clear

load('umist.mat');
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
QAP_labels={};
tic;
for i=1:10
    i
    
    [y,x]=kernel(initialx);
    y=y(y~=0);
    l=mat2label(x',n);
    [NMI,AC]=ACNMI(l,label);
    QAP(i,1)=NMI;
    QAP(i,2)=AC;
     QAP(i,3)=y(end);
     QAP_labels{i}=l;
end



for i=1:10
    [l_BKmeans]=balanced_kmeans(d,nClass);
    [NMI_BKmeans,AC_BKmeans]=ACNMI(l_BKmeans,label);
%   [S_BKmeans,class]=objective(fea,l_BKmeans,n);
    x_BKmeans(i,:)=[NMI_BKmeans,AC_BKmeans];
    BKmeans_labels{i}=l_BKmeans;
end


x_ckmeans=[];
CKmeans_labels={};
for i=1:10
    ckmean(d,nClass);
    load('l4.mat');
    l_ckmeans=l4;
    l_ckmeans=double(l_ckmeans);
    l_ckmeans=l_ckmeans';
    [NMI_ckmeans,AC_ckmeans]=ACNMI(l_ckmeans,label);
    [S_ckmeans,class]=objective(fea,l_ckmeans,n);
    x_ckmeans=[x_ckmeans;NMI_ckmeans,AC_ckmeans,S_ckmeans];
    CKmeans_labels{i}=l_ckmeans;
end
x_QAP=QAP;
NMI_QAP_avg=mean(x_QAP(:,1)*100);
NMI_QAP_max=max(x_QAP(:,1)*100);
NMI_QAP_min=min(x_QAP(:,1)*100);
NMI_QAP_std=std(x_QAP(:,1)*100);
AC_QAP_avg=mean(x_QAP(:,2)*100);
AC_QAP_max=max(x_QAP(:,2)*100);
AC_QAP_min=min(x_QAP(:,2)*100);
AC_QAP_std=std(x_QAP(:,2)*100);

NMI_BKmeans_avg=mean(x_BKmeans(:,1)*100);
NMI_BKmeans_max=max(x_BKmeans(:,1)*100);
NMI_BKmeans_min=min(x_BKmeans(:,1)*100);
NMI_BKmeans_std=std(x_BKmeans(:,1)*100);
AC_BKmeans_avg=mean(x_BKmeans(:,2)*100);
AC_BKmeans_max=max(x_BKmeans(:,2)*100);
AC_BKmeans_min=min(x_BKmeans(:,2)*100);
AC_BKmeans_std=std(x_BKmeans(:,2)*100);

NMI_ckmeans_avg=mean(x_ckmeans(:,1)*100);
NMI_ckmeans_max=max(x_ckmeans(:,1)*100);
NMI_ckmeans_min=min(x_ckmeans(:,1)*100);
NMI_ckmeans_std=std(x_ckmeans(:,1)*100);
AC_ckmeans_avg=mean(x_ckmeans(:,2)*100);
AC_ckmeans_max=max(x_ckmeans(:,2)*100);
AC_ckmeans_min=min(x_ckmeans(:,2)*100);
AC_ckmeans_std=std(x_ckmeans(:,2)*100);

save labels_umist.mat d fea QAP_labels BKmeans_labels CKmeans_labels
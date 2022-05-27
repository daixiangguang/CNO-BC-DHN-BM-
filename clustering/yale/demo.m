clc;
clear;
load('yale.mat');
if ~exist('label','var')
    label=gnd;
end
nClass=max(unique(label));
fea=normlizedata(fea,2);
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
k=1;
% QAP
    d=gaussinKernel(fea,4);%4
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


for i=1:5
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
time=toc;
temp=temp(temp~=0);
temp=temp/k;
x_QAP=QAP;
%GBC
y_GBC=[];
GBC_labels={};
parfor i=1:20
    [l_GBC]=GBC(fea,nClass,1,0.001,1e-5,200);
    [NMI_GBC,AC_GBC]=ACNMI(l_GBC,label);
    %[S_GBC,class]=objective(fea,l_GBC,n);
    x_GBC(i,:)=[NMI_GBC,AC_GBC];
    GBC_labels{i}=l_GBC;
    y_GBC(i,:)=l_GBC;
end
QAP_labels={};
tic;
y_BKmeans=[];
for i=1:20
    [l_BKmeans]=balanced_kmeans(fea,nClass);
    [NMI_BKmeans,AC_BKmeans]=ACNMI(l_BKmeans,label);
%   [S_BKmeans,class]=objective(fea,l_BKmeans,n);
    x_BKmeans(i,:)=[NMI_BKmeans,AC_BKmeans];
    BKmeans_labels{i}=l_BKmeans;
    y_BKmeans(i,:)=l_BKmeans;
end

% % ckmeans
x_ckmeans=[];
CKmeans_labels={};
y_ckmeans=[];
for i=1:20
    ckmean(fea,nClass);
    load('l4.mat');
    l_ckmeans=l4;
    l_ckmeans=double(l_ckmeans);
    l_ckmeans=l_ckmeans';
    [NMI_ckmeans,AC_ckmeans]=ACNMI(l_ckmeans,label);
    [S_ckmeans,class]=objective(fea,l_ckmeans,n);
    x_ckmeans=[x_ckmeans;NMI_ckmeans,AC_ckmeans,S_ckmeans];
    CKmeans_labels{i}=l_ckmeans;
    y_ckmeans(i,:)=l_ckmeans;
end




save result.mat x_BKmeans x_ckmeans x_QAP x_GBC

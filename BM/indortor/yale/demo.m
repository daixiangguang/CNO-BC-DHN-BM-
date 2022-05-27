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


k=5;
d=gaussinKernel(fea,4);
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
NMI_QAP_avg=mean(x_QAP(:,1)*100);
NMI_QAP_max=max(x_QAP(:,1)*100);
NMI_QAP_min=min(x_QAP(:,1)*100);
NMI_QAP_std=std(x_QAP(:,1)*100);
AC_QAP_avg=mean(x_QAP(:,2)*100);
AC_QAP_max=max(x_QAP(:,2)*100);
AC_QAP_min=min(x_QAP(:,2)*100);
AC_QAP_std=std(x_QAP(:,2)*100);

save result4.mat x_QAP
save 8.mat QAP_labels d fea
% 0.1 0.1 50 100 32 result labels_yale
%0.1 0.1 50 200 32*2 result2 labels_yale2
%0.3 0.2 200 200 32 result3 labels_yale3
%1 0.1 200 200 32 
%1 0.2 200 200 32 
%0.8 0.2 200 200 32 
%0.8 0.5 200 200 32 
%1 0.5 200 200 32 
%5 0.5 200 200 32*4  ²î
%10 0.5 200 200 32*4  ²î
%10 0.9 200 200 32*4  ²î
%10 0.1 300 200 32  ²î
%1 0.9 300 200 32  ²î
clc
clear
load('Yale.mat');
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
k=10;
% QAP
d=gaussinKernel(fea,4);
d=-d;
d=d-diag(diag(d));

temp=0;
tic;
% y=cell(k,1);
pop=32;
p=max(label);
initialx=zeros(n*p,pop);
rng(1);
for i=1:pop
    temp=randn(n*p,1);
    temp(temp>=0)=1;
    temp(temp<0)=-1;
    initialx(:,i)=temp;
end
initialx=initialx';
M=[10 50 100 200];
if pop==1
    y1=zeros(1,10);
else
    y1=zeros(4,10);
end
y_N_32=cell(1,1);
for it=1:10
    it
    [y,x]=kernel(initialx,300);
    y=y(y~=0);
    y_N_32{it}=y;
    if pop==1
        
    y1(1,it)=find_unstop_point(y,1);
    else
        
    y1(1,it)=find_unstop_point(y,M(1));
    y1(2,it)=find_unstop_point(y,M(2));
    y1(3,it)=find_unstop_point(y,M(3));
    y1(4,it)=find_unstop_point(y,M(4));
    end
end

%{
for it=1:10
    y1(1,it)=find_unstop_point(y_N_32{it},M(1));
    y1(2,it)=find_unstop_point(y_N_32{it},M(2));
    y1(3,it)=find_unstop_point(y_N_32{it},M(3));
    y1(4,it)=find_unstop_point(y_N_32{it},M(4));
end
%}
   
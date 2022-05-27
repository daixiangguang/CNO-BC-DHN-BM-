%% I. ������ʼ��
clc
clear
close all
load('umist.mat');
p=20;
fea=normlizedata(fea,1);
%fea=zscore(fea);
[n,m]=size(fea);
options.ReducedDim=fix(0.9*m);
W = PCA(fea,options);
fea=fea*W;
[n,~]=size(fea);

%QAPPSOHopfiledSyn 

options1.cores=1;
%options1.N=1;
options1.alpha=single(gpuArray(15));
options1.beta=single(gpuArray(15));
options1.iterations=20000;
options1.p=p;
   d=gaussinKernel(fea,0.2);
   d=-d;
   d=d-diag(diag(d));
   fea=d;




%�޸����´���
options1.N=96; %��Ⱥ 
pop=options1.N;
options1.pop=pop;
M=[10 100 200 300];
if pop==1
    y1=zeros(1,10);
else
    y1=zeros(4,10);
end

for it=1:10
    it
    y=QAPPSOHopfieldPal(fea,options1,500);
    y=y(y~=0);
    y=gather(y);
    y=double(y);
    result{it}=y;
    if pop==1
        
    y1(1,it)=find_unstop_point(y,1);
    else
        
    y1(1,it)=find_unstop_point(y,M(1));
    y1(2,it)=find_unstop_point(y,M(2));
    y1(3,it)=find_unstop_point(y,M(3));
    y1(4,it)=find_unstop_point(y,M(4));
    end
end


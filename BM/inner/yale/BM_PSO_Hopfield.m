function [x,obj] = BM_PSO_Hopfield(X,options)
% get parameter 
[n,~]=size(X);
pop=options.pop;
alpha=options.alpha;
beta=options.beta;
p=options.p;
d=X; 
u=ceil(n/p);
maxiter=options.iterations;
C1=options.C1;
C2=options.C2;
stopNum=500;
iter=2000;
pbest_x=[];
% set init_x(hopfiled x) and init_v(update speed and location)
for i=1:pop
    ran=randn(n,p);
    ran(ran>=0)=1;
    ran(ran<0)=-1;
    initial_x{i}=ran;
    pbest_x=[pbest_x,ran];
end
x=cell(pop,1);
%initial_x=randn(n,p*pop);
%initial_x(initial_x>=0)=1;
%initial_x(initial_x<0)=-1;
initial_v=zeros(n,p*pop);

pbest=ones(pop,1)*inf; % local optimum
y=zeros(maxiter,1); % function value
I1=ones(n,n); %the sum of each row ÐÐ
I2=zeros(pop*p,pop*p); %the sum of each column ÁÐ
block=ones(p,p);
%pbest_x=initial_x;
for i=1:pop
    I2((i-1)*p+1:p*i,(i-1)*p+1:p*i)=block;
end
one=eye(n,n);
const1=0.5*d+beta*(I1-one);
one=eye(p*pop,p*pop);
const2=alpha*(I2-one);
const3=repmat(alpha*(-2+p)+beta*(-2*u+n),n,p*pop);
zbest=inf;
flag=ones(1,pop);
tx=[];

[x,obj]=BM_HopfieldParallel1(initial_x{1},d,p,beta,alpha,iter,pop,n);



function [x] = BM_HopfieldParallel2(x,d,p,beta,alpha,iter,pop,n)
it=0;
while(it<iter)
    rd=randperm(n*p);
    T=T0*0.9^(it);
    for index=1:n*p
        [i,k]=ind2sub(size(x),rd(index));
        dedx=0.5*d(i,:)*x(:,k)+beta*(sum(x(i,:))-2+p-x(i,k))+alpha*(sum(x(:,k))-2*n/p+n-x(i,k));
        dedx=-dedx;
        pb=1/(1+exp(-dedx/T));
        if pb<rand
            x(i,k)=-1;
        else
            x(i,k)=1;
        end
    end
  it=it+1;
end


function [x,obj] = BM_HopfieldParallel1(x,d,p,beta,alpha,iter,pop,n)
result=zeros(n,p);
for index=1:n*p
      [i,k]=ind2sub(size(x),index);
      result(i,k)=0.5*d(i,:)*x(:,k)+beta*(sum(x(i,:))-2+p-x(i,k))+alpha*(sum(x(:,k))-2*n/p+n-x(i,k));
end
T0=mean(abs(result(:)));
it=0;
obj=[];
while(it<iter)
    rd=randperm(n*p);
    T=T0*0.99^(it);
    for index=1:n*p
        [i,k]=ind2sub(size(x),rd(index));
        dedx=0.5*d(i,:)*x(:,k)+beta*(sum(x(i,:))-2+p-x(i,k))+alpha*(sum(x(:,k))-2*n/p+n-x(i,k));
        dedx=-dedx;
        pb=1/(1+exp(-dedx/T));
        if pb<rand
            x(i,k)=-1;
        else
            x(i,k)=1;
        end
    end
    it=it+1;
    f=sum(diag(0.5*x'*d*x))+0.5*beta*sum((sum(x,2)-2+p).*(sum(x,2)-2+p))+0.5*beta*sum((sum(x,1)-2*n/p+n).*(sum(x,1)-2*n/p+n));
    obj=[obj;f];
end




function [x] = BM_HopfieldParallel(x,const1,const2,const3,iter)
[m,n]=size(x);
I=ones(m,n);
dedx=const1*x+x*const2+const3;
T0=mean(abs(dedx(:)));
for t=0:iter
    T=T0*0.9^(t);
    dedx=const1*x+x*const2+const3;
    pb=I./(1+exp(dedx/T));
    for i = 1 :m*n
        if pb(i)<rand
            x(i)=-1;
        else
            x(i)=1;
        end
    end
end

%Stopping conditions
function flag=stop(y,i,k)
flag=0;
if i>=k
    if length(unique(y(i-k+1:i)))==1
        flag=1;
    end
end


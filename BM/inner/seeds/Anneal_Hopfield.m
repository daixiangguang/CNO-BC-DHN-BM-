function [x,f]=Anneal_Hopfield(n,d,p,B,x) 
it=0;
% 求解目标函数值
dedx_new=zeros(n,p);
obj=[];
for index=1:n*p
      [i,k]=ind2sub(size(x),index);
      dedx_new(i,k)=0.5*d(i,:)*x(:,k)+B*(sum(x(i,:))-2+p-x(i,k))+B*(sum(x(:,k))-2*n/p+n-x(i,k));
end
T0=mean(abs(dedx_new(:)));
while (it<2000)
        rd=randperm(n*p);
        T=T0*0.9^(it);
        for index=1:n*p
            [i,k]=ind2sub(size(x),rd(index));
            dedx_new=0.5*d(i,:)*x(:,k)+B*(sum(x(i,:))-2+p-x(i,k))+B*(sum(x(:,k))-2*n/p+n-x(i,k));
            dedx_new=-dedx_new;
            pb=1/(1+exp(-dedx_new/T));
            if pb<rand
                x(i,k)=-1;
            else
                x(i,k)=1;
            end
        end
        it=it+1;
end
f=sum(diag(0.5*x'*d*x))+0.5*B*sum((sum(x,2)-2+p).*(sum(x,2)-2+p))+0.5*B*sum((sum(x,1)-2*n/p+n).*(sum(x,1)-2*n/p+n));




  

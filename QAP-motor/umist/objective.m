function [S,class]=objective(X,l,n)
% n��ʾ��������
% k��ʾ��
u=unique(l);%ɾ���ظ��ı�ǩ
class=[];
for j=1:length(u)
    t=[];
    for i=1:n
        if l(i)==u(j)
            t=[t i];
        end
    end
    class=[class;t];
end
[c1,c2]=size(class);
S=0;
d=squareform(pdist(X)); 
for i=1:c1
    temp=0;
    for j=1:c2
        for o=j+1:c2
            index1=class(i,j);
            index2=class(i,o);
            temp=temp+d(index1,index2);
        end
    end
    S=S+temp;
end

% for i=1:c1
%     for j=1:c2-1
%         for o=j+1:c2
% %             S=S+norm(X(class(i,j),:)-X(class(i,o),:),2)^2;
%             S=S+(X(class(i,j),:)-X(class(i,o),:))*(X(class(i,j),:)-X(class(i,o),:))';
%         end
%     end
% end

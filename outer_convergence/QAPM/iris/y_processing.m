function yb=y_processing(y)
l=find(y(end)==y);%�ҵ���һ��������
tail=ceil(l(1)*0.3);
yb=zeros(l(1)+tail,1);
yb(1:l)=y(1:l);
yb(l+1:end)=repmat(y(l(1)),tail,1);
total=0;
xx=randn(5,4);
xx(xx>=0)=1;
xx(xx<0)=-1;
tt=rand(5,4);
for i=1:20
    if xx(i)==1
    total=total+tt(i);
    else
        total=total-tt(i);
    end
end

res1=0.5*total;

total=0;
for i=1:20
    if xx(i)==1
    total=total+0.5*tt(i);
    else
        total=total-0.5*tt(i);
    end
end

res2=total;
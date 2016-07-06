function [x,n]=Jacbio(a,d,x0)
stop=1.0e-4; %迭代的精度?
m=size(a,1);
L=-tril(a,-1);
U=-triu(a,1);
D=diag(diag(a))\eye(m);
X=D*(L+U)*x0+D*d; %?J迭代公式?
n=1;
while norm(X-x0,inf)>=stop %?时迭代中止否则继续?
    x0=X;
X=D*(L+U)*x0+D*d;
n=n+1;
end
x=X;
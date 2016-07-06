function [x,n]=Jacbio(a,d,x0)
stop=1.0e-4; %�����ľ���?
m=size(a,1);
L=-tril(a,-1);
U=-triu(a,1);
D=diag(diag(a))\eye(m);
X=D*(L+U)*x0+D*d; %?J������ʽ?
n=1;
while norm(X-x0,inf)>=stop %?ʱ������ֹ�������?
    x0=X;
X=D*(L+U)*x0+D*d;
n=n+1;
end
x=X;
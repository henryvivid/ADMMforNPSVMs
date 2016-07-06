function [x, niters] = CGsolver(TH,TT,b,ecur,rho)
% CGsolver : Solve Ax=b by conjugate gradients
%
% Given symmetric positive definite sparse matrix A and vector b,
% this runs conjugate gradient to solve for x in A*x=b.
% It iterates until the residual norm is reduced by 10^-6,
% or for at most max(100,sqrt(n)) iterations
[mth,~]=size(TH);[mtt,~]=size(TT);
n = length(b);

tol = 1e-4;
%maxiters = max(100,sqrt(n));
maxiters = 50;
normb = norm(b);
x = zeros(n,1);
r = b;
rtr = r'*r;
d = r;
niters = 0;
while sqrt(rtr)/normb > tol  &&  niters < maxiters
    niters = niters+1;
    %%Ad = A*d;
    %Acon=[TH; -TH; -TT];
    d12=d(1:mth)-d(mth+1:2*mth);
    d3=d(2*mth+1:2*mth+mtt);
    A1=TH'*d12-TT'*d3;
    H1=A1'*A1;
    A2=ecur'*d;H21=A2'*A2;
    H22=d'*d;
    H2=rho*(H21+2*H22);
    H=H1+H2;
    %alpha = rtr / (d'*Ad);
    alpha = rtr / H;
    
    x = x + alpha * d;
    %r = r - alpha * Ad;
    Ad1=TH*A1;Ad3=TT*A1;
    Ad=[Ad1;-Ad1;-Ad3]+rho*(ecur*A2+2*d);
    r = r - alpha * Ad;
    
    rtrold = rtr;
    rtr = r'*r;
    beta = rtr / rtrold;
    d = r + beta * d;
end
end
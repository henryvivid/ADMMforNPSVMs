function [wp,bp,wn,bn,sv]=ADMMforNPSVMprimal(traindata, trainlabel,  Cvec, epsilon, rho)
% ADMM algorithm for NPSVM primal problems
%'traindata' is a training data matrix , each row is a sample vector
%'trainlabel' is a label vector,should  start  from 1 to p+q

% parameters predefined and initialization
TX=traindata;TY=trainlabel;
%[m,~] = size(TX);

TA=TX(TY==1,:);TB=TX(TY==-1,:);
%LA=TY(TY==1,:);LB=TY(TY==-1,:);
[p,~]=size(TA);[q,~]=size(TB);
clear TX;clear TY;

ep=ones(p,1);en=ones(q,1);

c1=Cvec;c2=Cvec;c3=Cvec;c4=Cvec;

kvp=[c1*ep' c1*ep' c2*en']';
ccp=[-epsilon*ep' -epsilon*ep' en']';

kvn=[c3*en' c3*en' c4*ep']';
ccn=[-epsilon*en' -epsilon*en' ep']';

[wp,bp]=ADMM_im(TA,TB,kvp,ccp,rho,1);
[wn,bn]=ADMM_im(TB,TA,kvn,ccn,rho,2);

sv1=p-length(find(TA*wp+bp>=epsilon&TA*wn+bn<=1));
sv2=q-length(find(TB*wn+bn>=epsilon&TB*wp+bp>=-1));  
sv=(sv1+sv2)/(p+q);

end

%% ADMM for NPSVM primal problem with matrix inversion
function [w,b]=ADMM_im(thisdata,otherdata,kv,cc,rho,indicate)

%[mh,~]=size(H);
TH=thisdata;TT=otherdata;
[mta,n]=size(TH);[mtb,~]=size(TT);mh=2*mta+mtb;
TH=[TH ones(mta,1)];TT=[TT ones(mtb,1)];
P=zeros(n+1,1);

L=zeros(mh,1);
U=zeros(length(cc),1);

%Global constants and defaults
MAX_ITER = 100;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

rval=zeros(MAX_ITER,1);sval=zeros(MAX_ITER,1);%fval=zeros(MAX_ITER,1);
eps_pri=zeros(MAX_ITER,1);eps_dual=zeros(MAX_ITER,1);

if indicate==2
   TT=-TT;
end
CA=[-TH;TH;-TT];
PM = invChol_mex(eye(n+1)+rho*(CA'*CA));

k=1;
while k <=MAX_ITER
    Lold=L;
    
    %V = U + CB*L - cc;                          %V = U+B*Gamma-c
    CBL=L;
    V = U + CBL - cc;
    
    %br = - qv - rho * CA'*V;   % br is  vector b in AX=b for CG
    CAtV=CA'*V;
    br = - rho * CAtV;

    P=PM*br;
    
    % update L
    %L = CB'*(cc-CA*P-U);
    CAP=CA*P;
    Ltemp = (cc-CA*P-U);theta=kv/rho;
    L(Ltemp>theta)=Ltemp(Ltemp>theta)-theta(Ltemp>theta);
    L(Ltemp>0&Ltemp<theta)=0;
    L(Ltemp<0)=Ltemp(Ltemp<0);   
    
    % update U
    %r = CA*P + CB*L - cc;  % r= A*Pi+B*Gamma, used for the primal residual
    r = CAP + L - cc;
    U = U + r;
    
    %fval(k)=0.5*P'*H*P+qv'*P;
    %Acon=[TH;-TH;-TT];F1=Acon'*P;
    %fval(k)=0.5*(F1'*F1)+qv'*P;
    
    %s = rho*CA'*(CB*(L- Lold));
    Lup=L- Lold;
    Lup12=Lup(mta+1:2*mta)-Lup(1:mta);
    s=rho*(TH'*Lup12-TT'*Lup(2*mta+1:mh));
    
    rval(k)  = norm(r); %record the primal residual at k
    sval(k)  = norm(s); %record the dual residual at k
    
    %an absolute tolerance xi^abs  and a relative tolerance xi^rel
    CAtU=CA'*U;
    eps_pri(k) = sqrt(mh+2)*ABSTOL + RELTOL*max([norm(CAP), norm(L), norm(cc)]);
    eps_dual(k)= sqrt(2*mh)*ABSTOL + RELTOL*norm(rho*CAtU);      
    
    if  rval(k) < eps_pri(k) && sval(k) < eps_dual(k);
        break
    end
    
    k=k+1;
end

w=P(1:n);b=P(n+1);

end


%% ADMM for NPSVM primal problem with conjugate gradient algortihm
function [w,b]=ADMMp(thisdata,otherdata,kv,cc,rho,indicate)

%[mh,~]=size(H);
TH=thisdata;TT=otherdata;
[mta,n]=size(TH);[mtb,~]=size(TT);mh=2*mta+mtb;
TH=[TH ones(mta,1)];TT=[TT ones(mtb,1)];
P=zeros(n+1,1);

L=zeros(mh,1);
U=zeros(length(cc),1);
%Global constants and defaults
MAX_ITER = 10;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

k=1;rval=zeros(MAX_ITER,1);sval=zeros(MAX_ITER,1);%fval=zeros(MAX_ITER,1);
eps_pri=zeros(MAX_ITER,1);eps_dual=zeros(MAX_ITER,1);
if indicate==2
   TT=-TT;
end

CA=[-TH;TH;-TT];

while k <=MAX_ITER
    Lold=L;
    
    %V = U + CB*L - cc;                          %V = U+B*Gamma-c
    CBL=L;
    V = U + CBL - cc;
    
    %br = - qv - rho * CA'*V;   % br is  vector b in AX=b for CG
    br = - rho * CA'*V;
    
    
    %[P, ~] = CGsolver(HH,br);  %CG mothed to solve a linear equations Pi
    [P, ~] = CG_NPSVMprimal(TH,TT,br,rho);

    % update L
    Ltemp = (cc-CA*P-U);theta=kv/rho;
    if Ltemp>theta
        L=Ltemp-theta;
    elseif Ltemp>0&Ltemp<theta
        L=Ltemp+theta;
    elseif Ltemp<0
        L=0;
    end
    
    % update U
    %r = CA*P + CB*L - cc;  %r= A*Pi+B*Gamma, used for the primal residual
    r = CA*P + L - cc;
    U = U + r;
    
    %fval(k)=0.5*P'*H*P+qv'*P;
    
    %s = rho*CA'*(CB*(L- Lold));
    Lup=L- Lold;
    s=rho*CA'*Lup;
    
    rval(k)  = norm(r); %record the primal residual at k
    sval(k)  = norm(s); %record the dual residual at k
    
    %an absolute tolerance xi^abs  and a relative tolerance xi^rel
    CAtU=CA'*U;
    eps_pri(k) = sqrt(mh+2)*ABSTOL + RELTOL*max([norm(CA*P), norm(CBL), norm(cc)]);
    eps_dual(k)= sqrt(2*mh)*ABSTOL + RELTOL*norm(rho*CAtU);      
    
    if  rval(k) < eps_pri(k) && sval(k) < eps_dual(k);
        break
    end
    
    k=k+1;
end

w=P(1:n);b=P(n+1);
end

function [x, niters] = CG_NPSVMprimal(TH,TT,b,rho)
% CGsolver : Solve Ax=b by conjugate gradients
%
% Given symmetric positive definite sparse matrix A and vector b,
% this runs conjugate gradient to solve for x in A*x=b.
% It iterates until the residual norm is reduced by 10^-6,
% or for at most max(100,sqrt(n)) iterations
%[mth,~]=size(TH);[mtt,~]=size(TT);

nb = length(b);
CA=[-TH;TH;-TT];

tol = 1e-4;
%maxiters = max(100,sqrt(n));
maxiters = 50;
normb = norm(b);
x = zeros(nb,1);
r = b;
rtr = r'*r;
d = r;
niters = 0;
while sqrt(rtr)/normb > tol  &&  niters < maxiters
    niters = niters+1;
    %%Ad = A*d; %% A denotes the matrix in Ax=b;In ADMM, A=H+rho*CA'*CA;
    CAd=[-TH*d;TH*d;-TT*d];
    H1=d'*d;H2=rho*(CAd'*CAd);
    H=(H1+H2);    

    %alpha = rtr / (d'*Ad);
    alpha = rtr / H; %% H=d'*A*d;
    
    x = x + alpha * d;
    %r = r - alpha * Ad;
    Ad=d+rho*CA'*CAd;
    r = r - alpha * Ad;
    
    rtrold = rtr;
    rtr = r'*r;
    beta = rtr / rtrold;
    d = r + beta * d;
end
end



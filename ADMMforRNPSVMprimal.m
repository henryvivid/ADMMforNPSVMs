function [wp,bp,wn,bn,sv]=ADMMforRNPSVMprimal(traindata, trainlabel,  Cvec, epsilon, tband, sband, rho)
% ADMM algorithm for RNPSVM primal problems
%'traindata' is a training data matrix , each row is a sample vector
%'trainlabel' is a label vector,should  start  from 1 to p+q

% parameters predefined and initialization
TX=traindata;TY=trainlabel;
%[m,~] = size(TX);
max_iter=50;

TA=TX(TY==1,:);TB=TX(TY==-1,:);
LA=TY(TY==1,:);LB=TY(TY==-1,:);
[p,~]=size(TA);[q,~]=size(TB);
clear TX;clear TY;

ep=ones(p,1);en=ones(q,1);
delta_p0=zeros(p,1);delta_n0=zeros(q,1);
theta_str_p0=zeros(p,1);theta_str_n0=zeros(q,1);
theta_cur_p0=zeros(p,1);theta_cur_n0=zeros(q,1);

for kp=1:max_iter
    
    tp_old=[delta_p0' theta_str_n0' theta_cur_n0'];
    tn_old=[delta_n0' theta_str_p0' theta_cur_p0'];
    
    c1=Cvec;c2=Cvec;c3=Cvec;c4=Cvec;
    
    qvp=[TA'*(theta_str_p0-theta_cur_p0)+TB'*(delta_n0.*LB);...
        sum(theta_str_p0-theta_cur_p0)+sum(delta_n0.*LB)];
    kvp=[c1*ep' c1*ep' c2*en']';
    ccp=[-epsilon*ep' -epsilon*ep' en']';
    
    qvn=[TB'*(theta_str_n0-theta_cur_n0)+TA'*(delta_p0.*LA);...
        sum(theta_str_n0-theta_cur_n0)+sum(delta_p0.*LA)];
    kvn=[c3*en' c3*en' c4*ep']';
    ccn=[-epsilon*en' -epsilon*en' ep']';
    
    %tic
    [wp,bp]=ADMM_im(TA,TB,qvp,kvp,ccp,rho,1);
    [wn,bn]=ADMM_im(TB,TA,qvn,kvn,ccn,rho,2);
    %toc
    
    dp0=LA.*(TA*wn+bn)<sband;delta_p0(dp0==1)=c4;delta_p0(dp0==0)=0;
    dn0=LB.*(TB*wp+bp)<sband;delta_n0(dn0==1)=c2;delta_n0(dn0==0)=0;
    
    tsp0=TA*wp+bp<-tband;theta_str_p0(tsp0==1)=c1;theta_str_p0(tsp0==0)=0;
    tsn0=TB*wn+bn<-tband;theta_str_n0(tsn0==1)=c3;theta_str_n0(tsn0==0)=0;
    tcp0=TA*wp+bp>tband;theta_cur_p0(tcp0==1)=-c1;theta_cur_p0(tcp0==0)=0;
    tcn0=TB*wn+bn>tband;theta_cur_n0(tcn0==1)=-c3;theta_cur_n0(tcn0==0)=0;
    
    tp_new=[delta_p0' theta_str_n0' theta_cur_n0'];
    tn_new=[delta_n0' theta_str_p0' theta_cur_p0'];
    
    sv1=p-length(find(TA*wp+bp>=epsilon&TA*wn+bn<=1));
    sv2=q-length(find(TB*wn+bn>=epsilon&TB*wp+bp>=-1));
    
    sv=(sv1+sv2)/(p+q);
    
    if norm(tp_old-tp_new)<1e-4&&norm(tn_old-tn_new)<1e-4
        break;
    end
end
%disp(['The final CCCP Iteration is ',num2str(kp)]);
end

%% ADMM for NPSVM primal problem with matrix inversion
function [w,b]=ADMM_im(thisdata,otherdata,qv,kv,cc,rho,indicate)

%[mh,~]=size(H);
TH=thisdata;TT=otherdata;
[mta,n]=size(TH);[mtb,~]=size(TT);mh=2*mta+mtb;
TH=[TH ones(mta,1)];TT=[TT ones(mtb,1)];
P=zeros(n+1,1);

L=zeros(mh,1);
U=zeros(length(cc),1);

%Global constants and defaults
MAX_ITER = 50;
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
    br = qv- rho * CAtV;
    
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
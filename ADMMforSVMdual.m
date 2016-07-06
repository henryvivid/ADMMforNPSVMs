function [w,b,sv]=ADMMforSVMdual(traindata, trainlabel,  Cvec, rho)
% ADMM algorithm for SVM dual problems
%'traindata' is a training data matrix , each row is a sample vector
%'trainlabel' is a label vector,should  start  from 1 to p+q

% parameters predefined and initialization
TX=traindata;TY=trainlabel;
[m,~] = size(TX);

et=ones(m,1);
c=Cvec;

V=zeros(m,1);
U=c*et;
cc=[0 V' U']';

qv=-et;econ=TY;

%tic
alpha=ADMM3(TX,TY,qv,econ,cc,rho);
%toc
w=(alpha.*TY)'*TX;w=w';

bs=find(alpha>0);
b=sum(TY(bs)-TX(bs,:)*w)/length(bs);

sv=length(bs)/m;
end

%% ADMM algortihm
function alpha=ADMM3(data,label,qv,econ,cc,rho)

%[mh,~]=size(H);
TX=data;TY=label;
[mt,n]=size(TX);
P=zeros(mt,1);

na=length(econ);
nb=2*na;L=zeros(nb,1);
U=zeros(length(cc),1);
%Global constants and defaults
MAX_ITER = 50;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

k=1;rval=zeros(MAX_ITER,1);sval=zeros(MAX_ITER,1);%fval=zeros(MAX_ITER,1);
eps_pri=zeros(MAX_ITER,1);eps_dual=zeros(MAX_ITER,1);
T=TX.*repmat(TY,1,n);
MA=[T';sqrt(rho)*econ'];

Q = invChol_mex(eye(n+1)+1/(2*rho)*(MA*MA'));
PM = Q*MA;
% M=PI-1/(2*rho)^2*MA'*P; 

while k <=MAX_ITER
    Lold=L;
    
    %V = U + CB*L - cc;                          %V = U+B*Gamma-c
    CBL=[0 -L(1:na)' L(na+1:2*na)']';
    V = U + CBL - cc;
    
    %br = - qv - rho * CA'*V;   % br is  vector b in AX=b for CG
    CAtV=econ*V(1)+V(2:na+1)+V(na+2:2*na+1);
    br = - qv - rho * CAtV;
    
    
    %[P, ~] = CGsolver(HH,br);  %CG mothed to solve a linear equations Pi
    %[P, ~] = CGsolver(TH,TT,br,ecur,rho);
    
    B=PM*br;
    P= 1/(2*rho)*br - 1/(2*rho)^2*MA'*B;
 %   P=M*br;
    
    % update L
    %L = CB'*(cc-CA*P-U);
    CAP=[econ'*P P' P']'; CAPtem=cc-CAP-U;
    L=[-CAPtem(2:mt+1)' CAPtem(mt+2:2*mt+1)']';
    L(L<0)=0;
    
    % update U
    %r = CA*P + CB*L - cc;  %r= A*Pi+B*Gamma, used for the primal residual
    CBL=[0 -L(1:na)' L(na+1:2*na)']';
    r = CAP + CBL - cc;
    U = U + r;
    
    %fval(k)=0.5*P'*H*P+qv'*P;
    %Acon=[TH;-TH;-TT];F1=Acon'*P;
    %fval(k)=0.5*(F1'*F1)+qv'*P;
    
    %s = rho*CA'*(CB*(L- Lold));
    Lup=L- Lold;CBLup=[0 -Lup(1:na)' Lup(na+1:2*na)']';
    CAt_CBLup=CBLup(2:na+1)+CBLup(na+2:2*na+1);
    s=rho*CAt_CBLup;
    
    rval(k)  = norm(r); %record the primal residual at k
    sval(k)  = norm(s); %record the dual residual at k
    
    %an absolute tolerance xi^abs  and a relative tolerance xi^rel
    CAtU=econ*U(1)+U(2:na+1)+U(na+2:2*na+1);
    eps_pri(k) = sqrt(mt+2)*ABSTOL + RELTOL*max([norm(CAP), norm(CBL), norm(cc)]);
    eps_dual(k)= sqrt(2*mt)*ABSTOL + RELTOL*norm(rho*CAtU);      
    
    if  rval(k) < eps_pri(k) && sval(k) < eps_dual(k);
        break
    end
    
    k=k+1;
end

%% plot the process of convergence
%   K = length(fval);
%     h = figure;
% plot(1:K, fval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
% ylabel('f(\pi^k)'); xlabel('iter (k)');
% 
% g = figure;
% subplot(2,1,1);
% semilogy(1:K, max(1e-8, rval), 'k', ...
%     1:K, eps_pri, 'k--',  'LineWidth', 2);
% ylabel('||r||_2');
% 
% subplot(2,1,2);
% semilogy(1:K, max(1e-8, sval), 'k', ...
%     1:K, eps_dual, 'k--', 'LineWidth', 2);
% ylabel('||s||_2'); xlabel('iter (k)');

alpha=P;
end



function [wp,bp,wn,bn,sv]=ADMMforNPSVMdual(traindata, trainlabel,  Cvec, epsilon, rho)
% ADMM algorithm for NPSVM dual problems
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
kcur=[epsilon*ep' epsilon*ep' -en']';
ecur=[-ep' ep' en']';
ktip=[epsilon*en' epsilon*en' -ep']';
etip=[-en' en' -ep']';

c1=Cvec;c2=Cvec;c3=Cvec;c4=Cvec;

Vp=zeros(2*p+q,1);
Up=[c1*ep' c1*ep' c2*en']';
ccp=[0 Vp' Up']';

Vn=zeros(2*q+p,1);
Un=[0 c3*en' c3*en' c4*ep']';
ccn=[Vn' Un']';
tic
[alpha_p1,alpha_p2,beta_n1]=ADMM(TA,TB,kcur,ecur,ccp,rho,1);
[alpha_n1,alpha_n2,beta_p1]=ADMM(TB,TA,ktip,etip,ccn,rho,2);
toc
wp=(alpha_p1-alpha_p2)'*TA-beta_n1'*TB;wp=wp';
wn=(alpha_n1-alpha_n2)'*TB+beta_p1'*TA;wn=wn';

bk=find(alpha_p2>0 & alpha_p2<c1);
bp=sum(epsilon-wp'*TA(bk,:)')/length(bk);

bs=find(alpha_n2>0&alpha_n2<c3);
bn=sum(epsilon-wn'*TB(bs,:)')/length(bs);

sv1=p-length(find(alpha_p1-alpha_p2<1e-4&beta_p1<1e-4));
sv2=q-length(find(alpha_n1-alpha_n2<1e-4&beta_n1<1e-4));
sv=(sv1+sv2)/(p+q)

end


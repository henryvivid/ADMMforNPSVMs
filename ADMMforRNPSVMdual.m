function [w1,b1,w2,b2,sv]=ADMMforRNPSVMdual(traindata, trainlabel,  Cvec, epsilon, tband, sband, rho)
% ADMM algorithm for RNPSVM dual problems
%'traindata' is a training data matrix , each row is a sample vector
%'trainlabel' is a label vector,should  start  from 1 to p+q

% parameters predefined and initialization
TX=traindata;TY=trainlabel;
%[m,~] = size(TX);
wp=0;bp=0;wn=0;bn=0;

max_iter=100;

TA=TX(TY==1,:);TB=TX(TY==-1,:);
LA=TY(TY==1,:);LB=TY(TY==-1,:);
[p,~]=size(TA);[q,~]=size(TB);
clear TX;clear TY;

ep=ones(p,1);en=ones(q,1);
kcur=[epsilon*ep' epsilon*ep' -en']';
ecur=[-ep' ep' en']';

ktip=[epsilon*en' epsilon*en' -ep']';
etip=[-en' en' -ep']';

c1=Cvec;c2=Cvec;c3=Cvec;c4=Cvec;

%AA=TA*TA';AB=TA*TB';BB=TB*TB';

delta_p0=zeros(p,1);delta_n0=zeros(q,1);
theta_str_p0=zeros(p,1);theta_str_n0=zeros(q,1);
theta_cur_p0=zeros(p,1);theta_cur_n0=zeros(q,1);

for kp=1:max_iter
    disp(['The CCCP Iteration is ',num2str(kp)]);
    tp_old=[delta_p0' theta_str_n0' theta_cur_n0'];
    tn_old=[delta_n0' theta_str_p0' theta_cur_p0'];
    
    Vp=[-theta_str_p0' theta_cur_p0' -delta_n0']';
    Up=[c1*ep'-theta_str_p0' c1*ep'+theta_cur_p0' c2*en'-delta_n0']';
    ccp=[0 Vp' Up']';
    
    Vn=[-theta_str_n0' theta_cur_n0' -delta_p0']';
    Un=[c3*en'-theta_str_n0' c3*en'+theta_cur_n0' c4*ep'-delta_p0']';
    ccn=[0 Vn' Un']';
    %tic
    [alpha_p1,alpha_p2,beta_n1]=ADMM(TA,TB,kcur,ecur,ccp,rho,1);
    [alpha_n1,alpha_n2,beta_p1]=ADMM(TB,TA,ktip,etip,ccn,rho,2);
    %toc
    wp=(alpha_p1-alpha_p2)'*TA-beta_n1'*TB;wp=wp';
    wn=(alpha_n1-alpha_n2)'*TB+beta_p1'*TA;wn=wn';
    
    bk=find(alpha_p2-theta_cur_p0>0 & alpha_p2-theta_cur_p0<c1);
%     bk2=find(alpha_p2-theta_cur_p0<c1);
%     bk=intersect(bk1,bk2);
    if isempty(bk)
        continue;
    end;
    %bk=bk(1);
    bp=sum(epsilon-wp'*TA(bk,:)')/length(bk);
    
    bs=find(alpha_n2-theta_cur_n0>0&alpha_n2-theta_cur_n0<c3);
    if isempty(bs)
        continue;
    end;
    %bs=bs(1);
    bn=sum(epsilon-wn'*TB(bs,:)')/length(bs);
    
    dp0=LA.*(TA*wn+bn)<sband;delta_p0(dp0==1)=c4;delta_p0(dp0==0)=0;
    dn0=LB.*(TB*wp+bp)<sband;delta_n0(dn0==1)=c2;delta_n0(dn0==0)=0;
    
    tsp0=TA*wp+bp<-tband;theta_str_p0(tsp0==1)=c1;theta_str_p0(tsp0==0)=0;
    tsn0=TB*wn+bn<-tband;theta_str_n0(tsn0==1)=c3;theta_str_n0(tsn0==0)=0;
    tcp0=TA*wp+bp>tband;theta_cur_p0(tcp0==1)=-c1;theta_cur_p0(tcp0==0)=0;
    tcn0=TB*wn+bn>tband;theta_cur_n0(tcn0==1)=-c3;theta_cur_n0(tcn0==0)=0;
    
    tp_new=[delta_p0' theta_str_n0' theta_cur_n0'];
    tn_new=[delta_n0' theta_str_p0' theta_cur_p0'];
    
    if norm(tp_old-tp_new)<1e-4&&norm(tn_old-tn_new)<1e-4
        break;
    end
end
    
sv1=p-length(find(alpha_p1-alpha_p2<1e-4&beta_p1<1e-4))
sv2=q-length(find(alpha_n1-alpha_n2<1e-4&beta_n1<1e-4))
sv=(sv1+sv2)/(p+q);
w1=wp;b1=bp;
w2=wn;b2=bn;
end


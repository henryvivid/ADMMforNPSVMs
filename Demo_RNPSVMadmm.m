tic;
clear;close all;clc;
format compact;

%% datasets
name1={'moon','shuttle','skin','a9a','w8a','ijcnn1','webspam','covtype',...
    'codrna','USPS','realsim','rcv1'};
name2={'WPBC','sonar','Spectf',... %% 3
    'heart','hungarian','heartc','bupa_liver','Ionosphere','dermatology','votes',... %% 7
    'Arrhythmia','clean1','WDBC','Australian','blood','pima','German','parkinson',...%% 8
    'iris','seeds','gem','wine','thyroid', 'circle','glass',... %% 7
    'vehicle','vowel','segment'}; %% 3
path0={'D:\mycodes\Metric_Learning\'};
pathsave = 'D:\mycodes\RNPSVM\Results\';
fsave = strcat(pathsave,'RNPSVMprimal_benchmark2','.xls');
tname=name2(1:28);
xlswrite(fsave,tname,1);
basenum=2;
gridcount=24;

re1=[];iter_run=5;v=5;
para_run=15;
for fi=1:18
    tic;
    name=name2{fi};
    disp(['The current runing dataset is ',name]);
    filename1= strcat(path0{1},name,'_scale.mat');
    DataName1=strcat(name,'_scale');
    S=load(filename1);
    EDX=S.(DataName1);[m,n]=size(EDX);
    rand('state',1);
    s=randperm(size(EDX,1));
    DX=EDX(s(1:m),:);clear EDX;
    
    Acc=zeros(1,para_run);err=zeros(1,para_run);st=zeros(1,para_run);
    for ic=1:para_run
        disp(['**************The parameter iteration is ',num2str(ic),'***********************']);
        c=basenum^(ic-7);
        for ts=1:gridcount
            disp(['The gridcount iteration is ',num2str(ts)]);
            tcount=ceil(ts/6);scount=ts-(tcount-1)*6;
            tband=0.4+(tcount-1)*0.2;
            sband=-1+(scount-1)*0.2;
            accuracy1=zeros(1,iter_run);err1=zeros(1,iter_run);
            for iter=1:iter_run
                %disp(['The crossvalidation iteration is ',num2str(iter)]);
                
                [TX,TY,EX,EY]=Crossvalidation(DX,v,iter);
                %TX=EX;TY=EY;
                [mt,nt]=size(TX);[me,ne]=size(EX);
                
                [my,ny]=hist(TY,unique(TY));K=length(my);label=zeros(me,K*(K-1)/2);
                itc=0;
                for ip=1:K-1
                    for jp=ip+1:K
                        itc=itc+1;
                        TXP=TX(TY==ny(ip),:);TYP=TY(TY==ny(ip),:);TYP=ones(length(TYP),1);
                        TXN=TX(TY==ny(jp),:);TYN=TY(TY==ny(jp),:);TYN=-ones(length(TYN),1);
                        TXS=[TXP;TXN];TYS=[TYP;TYN];
                        rho=1;epsilon=0.2;%tband=0.6;sband=0.8;
                        
                        %%%% seek for nonparallel hyperplanes
                        %[w1,b1,w2,b2,sv]=ADMMforRNPSVMdual(TXS, TYS,  c, epsilon, tband, sband, rho);
                        [w1,b1,w2,b2,sv]=ADMMforRNPSVMprimal(TXS, TYS,  c, epsilon, tband, sband, rho);
                        
                        dis1=abs(EX*w1+b1);dis2=abs(EX*w2+b2);
                        ind=dis1<dis2;
                        label(ind==1,itc)=ny(ip);label(ind==0,itc)=ny(jp);
                        
                    end
                end
                SV(iter)=sv;
                
                if itc==1
                    clabel=label;
                else
                    clabel=zeros(me,1);
                    for ie=1:me
                        a = sort(label(ie,:));
                        [cl,ia] = unique(a);
                        y = diff([ia;length(a)+1]);ys=sort(y,'descend');
                        if ys(1)==ys(2)
                            clabel(ie)=0;
                        else
                            clabel(ie)=cl(find(y==max(y),1));
                        end
                    end
                end
                
                accuracy1(iter)=sum(clabel==EY)/me*100;
                err1(iter)=100-accuracy1(iter);
            end
            Acc(ic,ts)=mean(accuracy1);
            err(ic,ts)=100-Acc(ic,ts);
            st(ic,ts)=std(err1);
            E(ic,ts).er=err1;
            
            D(ic,ts).wp=w1;D(ic,ts).bp=b1;
            D(ic,ts).wn=w2;D(ic,ts).bn=b2;
            SVs(ic,ts)=mean(SV);
            
        end
    end
    AC=max(max(Acc));
    [acx,acy]=find(Acc==AC,1);
    ERR=err(acx,acy)
    ST=st(acx,acy);
    svs=SVs(acx,acy)*100
    ler=E(acx,acy).er;
    toc;
    time=double(toc)/(iter_run*para_run*gridcount);
    disp(['The minimal error rate of ',name,' is ',num2str(ERR)]);
    disp(['The average CPU time of ',name,' is ',num2str(time)]);
    re1=[re1;acx acy ERR ST svs time 0 ler];
    
    resu=re1';
    xlswrite(fsave,resu,1,'A2');
end




%% SVM classification using LIBLINEAR(VS2008 is needed)

tic;
clear;close all;clc;
format compact;

%% datasets
name1={'moon','a9a','USPS','codrna','w8a','ijcnn1','webspam','covtype',...
    'shuttle','skin','rcv1','realsim'};
name2={'WPBC','sonar','Spectf',... %% 3
    'heart','hungarian','heartc','bupa_liver','Ionosphere','dermatology','votes',... %% 7
    'Arrhythmia','clean1','WDBC','Australian','blood','pima','German','parkinson',...%% 8
    'iris','seeds','gem','wine','thyroid', 'circle','glass',... %% 7
    'vehicle','vowel','segment'}; %% 3
path0={'D:\mycodes\Metric_Learning\'};
pathsave = 'D:\mycodes\RNPSVM\Results\';
fsave = strcat(pathsave,'SVM_liblinear','.xls');% The file to save the results
tname=name2(1:28);
xlswrite(fsave,tname,1);
basenum=2;

re1=[];para_run=15;iter_run=5; v=5;
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
        disp(['The parameter iteration is ',num2str(ic)]);
        c=ic-7;
        accur=zeros(1,iter_run);error=zeros(1,iter_run);
        for i=1:iter_run
            disp(['The crossvalidation iteration is ',num2str(i)]);
            [TD,TL,ED,EL]=Crossvalidation(DX,iter_run,i);
            TD=sparse(TD);ED=sparse(ED);
            TL=full(TL);EL=full(EL);
            
            cmd = ['-c ',num2str( basenum^c )];%
            model = train(TL,TD,cmd);
            [label,accuracy, dec_values] = predict(EL,ED,model);
            
            accur(i)=accuracy(1);
            error(i)=100-accur(i);
            %SV(i)=model.totalSV/length(TL);
        end
        Acc(ic)=mean(accur);
        err(ic)=mean(error);
        st(ic)=std(error);
        E(ic).er=error;
        %SVs(ic)=mean(SV);
    end
    toc;time=double(toc)/(iter_run*para_run);
    AC=max(Acc);
    acx=find(Acc==max(Acc),1);
    ERR=err(acx)
    ST=st(acx);
    %svs=SVs(acx)*100
    ler=E(acx).er;
    
    disp(['The minimal error rate of ',name,' is ',num2str(ERR)]);
    disp(['The average CPU time of ',name,' is ',num2str(time)]);
    re1=[re1;acx ERR ST svs time 0 ler];  
    
    resu=re1';
    xlswrite(fsave,resu,1,'A2');
end



%% SVM classification
% clear;
% close all;clc;

load segment_scale;
DX=segment_scale;
[m,n]=size(DX);
rand('state',2);
s=randperm(m);
DX=DX(s(1:m),:);

v=3;basenum=2;

for ic=-8:8
        for i=1:v
            [TD,TL,ED,EL]=Crossvalidation(DX,v,i);
            
            cmd = ['-s 0 -t 0',' -c ',num2str( basenum^ic )];%
            model = svmtrain(TL,TD,cmd);
            [label,accuracy] = svmpredict(EL,ED,model);

            accur(i)=accuracy(1);
			error(i)=100-accur(i);
        end
        AC(ic+9)=mean(accur);
        err(ic+9)=mean(error);
		st(ic+9)=std(error);
end

Acc=max(AC)
accx=find(AC==max(AC),1);
ERROR=err(accx);
STD=st(accx);

RESU=[acx-9 acy-9 100-Acc err];% sens prec

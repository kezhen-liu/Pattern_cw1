clear;

load('Q1_b_EigVec.mat');
load('Q1_b_DataSet.mat');

baseNum=364;
trainMean = mean(trainSet.').';
trainSetDiff = zeros(2576,364);
testSetDiff = zeros(2576,156);

principleEigvec=zeros(2576,baseNum);
% a=zeros(100,1);

ax=zeros(baseNum,364);
ay=zeros(baseNum,156);
confusion=zeros(3,52,52);

for i = 1:baseNum
    principleEigvec(:,i) = mEigVec(:,i);
end


for i = 1:364
    trainSetDiff(:,i) = trainSet(:,i)-trainMean;
    ax(:,i)=principleEigvec.'*trainSetDiff(:,i);
    %xbar(:,i)=principleEigvec*a + trainMean;
end

for i = 1:156
    testSetDiff(:,i) = testSet(:,i)-trainMean;
    ay(:,i)=principleEigvec.'*testSetDiff(:,i);
    %ybar(:,i)=principleEigvec*a + trainMean;
end

for l=1:3
    for m= 1:52 %trainset
        for n= 1:52 %test dataset
            for i= 1:7
                d(i)=norm(ax(:,i+(m-1)*7)-ay(:,l+(n-1)*3));
            end
            confusion(l,m,n)=min(d);
        end
    end
end
%--------------------------test---------------
%Columns 1 through 7

%     1.7067    2.5854    1.6418 

%m=3, l=1, n=1 
% the error between the first test face of the first person 
%  and the 7 faces of the third person
for i= 1:7
    d(i)=norm(ax(:,i+(3-1)*7)-ay(:,1+(1-1)*3))
end

%result: PCA decides that 
% 1st testface of the first person and the 7th trainface of the third person 
% are most alike
%???why!!!
%--------------------------test---------------

Minimum=zeros(3,52);
for l=1:3
  Minimum(l,:)=min(confusion(l,:,:));
end

NN=zeros(3,52,52);
for n= 1:52 %test dataset
    for l=1:3
        for m= 1:52 %trainset
            if confusion(l,m,n)> Minimum(l,n)
                NN(l,m,n)=0;
            else
                NN(l,m,n)=1;
            end
        end
    end
end

success=0;
for l=1:3
    for i=1:52
        success=success+NN(l,i,i);
    end
end

successrate=success/(3*52)
    

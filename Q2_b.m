clear;

TRAIN_NUM = 468;
TEST_NUM = 52;
EIGVEC_NUM = 467;

load('Q1_b_EigVec.mat');
load('Q1_b_DataSet.mat');

baseNum=EIGVEC_NUM;
trainMean = mean(trainSet.').';
trainSetDiff = zeros(2576,TRAIN_NUM);
testSetDiff = zeros(2576,TEST_NUM);

principleEigvec=zeros(2576,baseNum);
% a=zeros(100,1);

ax=zeros(baseNum,TRAIN_NUM);
ay=zeros(baseNum,TEST_NUM);
confusion=zeros(52,52,int32(TEST_NUM/52));

for i = 1:baseNum
    principleEigvec(:,i) = mEigVec(:,i);
end


for i = 1:TRAIN_NUM
    trainSetDiff(:,i) = trainSet(:,i)-trainMean;
    ax(:,i)=principleEigvec.'*trainSetDiff(:,i);
    %xbar(:,i)=principleEigvec*a + trainMean;
end

for i = 1:TEST_NUM
    testSetDiff(:,i) = testSet(:,i)-trainMean;
    ay(:,i)=principleEigvec.'*testSetDiff(:,i);
    %ybar(:,i)=principleEigvec*a + trainMean;
end

d=zeros(int32(TRAIN_NUM/52),1);
for l=1:int32(TEST_NUM/52)
    for m= 1:52 %trainset
        for n= 1:52 %test dataset
            for i= 1:int32(TRAIN_NUM/52)
                d(i)=norm(ax(:,i+(m-1)*int32(TRAIN_NUM/52))-ay(:,l+(n-1)*int32(TEST_NUM/52)));
            end
            confusion(m,n,l)=min(d);
        end
    end
end


Minimum=zeros(int32(TEST_NUM/52),52);
for l=1:int32(TEST_NUM/52)
  Minimum(l,:)=min(confusion(:,:,l));
end

NN=zeros(52,52,int32(TEST_NUM/52));
for n= 1:52 %test dataset
    for l=1:int32(TEST_NUM/52)
        for m= 1:52 %trainset
            if confusion(m,n,l)> Minimum(l,n)
                NN(m,n,l)=0;
            else
                NN(m,n,l)=1;
            end
        end
    end
end

success=0;
for l=1:int32(TEST_NUM/52)
    for i=1:52
        success=success+NN(i,i,l);
    end
end

successrate=success/TEST_NUM;

truthTable=diag(ones(52,1));

%plotconfusion(truthTable,NN);
    

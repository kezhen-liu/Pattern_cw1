clear;

load('Q1_b_EigVec.mat');
load('Q1_b_DataSet.mat');


trainMean = mean(trainSet.').';
trainSetDiff = zeros(2576,364);
testSetDiff = zeros(2576,156);

principleEigvec=zeros(2576,100);
% a=zeros(100,1);

xbar=zeros(2576,364);
ybar=zeros(2576,156);
confusion=zeros(3,52,52);

for i = 1:100
    principleEigvec(:,i) = mEigVec(:,i);
end


for i = 1:364
    trainSetDiff(:,i) = trainSet(:,i)-trainMean;
    a=principleEigvec.'*trainSetDiff(:,i);
    xbar(:,i)=principleEigvec*a + trainMean;
end

for i = 1:156
    testSetDiff(:,i) = testSet(:,i)-trainMean;
    a=principleEigvec.'*testSetDiff(:,i);
    ybar(:,i)=principleEigvec*a + trainMean;
end

for l=1:3
    for m= 1:52 %trainset
        for n= 1:52 %test dataset
            for i= 1:7
                d(i)=norm(xbar(:,i+(m-1)*7)-ybar(:,n+(l-1)*52));
            end
            confusion(l,m,n)=min(d);
        end
    end
end
        
        

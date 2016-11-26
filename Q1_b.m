%clear;

load('face.mat');

trainSet = zeros(2576, 364);
testSet = zeros(2576, 156);

testSetIndex = 1;
trainSetIndex = 1;
for i=1:520
    res = rem(i,10);
    if res > 7 || res == 0
        testSet(:,testSetIndex)=X(:,i);
        testSetIndex = testSetIndex + 1;
    else
        trainSet(:,trainSetIndex)=X(:,i);
        trainSetIndex = trainSetIndex + 1;
    end
end

S=cov(trainSet);
[V,D] = eig(S);

mD=zeros(364,1);

for i=1:364
    mD(i) = D(i,i);
end

mD=sort(mD);
        

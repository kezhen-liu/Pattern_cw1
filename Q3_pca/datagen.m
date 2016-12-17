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

trainEigVal=zeros(baseNum,TRAIN_NUM);
testEigVal=zeros(baseNum,TEST_NUM);
confusion=zeros(52,52,int32(TEST_NUM/52));

for i = 1:TRAIN_NUM
    trainSetDiff(:,i) = trainSet(:,i)-trainMean;
    trainEigVal(:,i)=mEigVec.'*trainSetDiff(:,i);
end

for i = 1:TEST_NUM
    testSetDiff(:,i) = testSet(:,i)-trainMean;
    testEigVal(:,i)=mEigVec.'*testSetDiff(:,i);
end

save('Q3_PCA_coeff','trainEigVal', 'testEigVal');
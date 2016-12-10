%clear;

%load('test_eigvec.mat');
load('Q1_b_EigVec.mat');
load('Q1_b_DataSet.mat');

trainMean = mean(trainSet.').';
trainSetDiff = zeros(2576,364);
for i = 1:364
    trainSetDiff(:,i) = trainSet(:,i)-trainMean;
end

rTrainSetDiff = trainSetDiff(:,1).' * mEigVec(:,1:364);

rImg =(rTrainSetDiff * mEigVec(:,1:364).').' + trainMean;

A=zeros(56,46);
for i=1:46
        A(:,i)=rImg(1+(i-1)*56:i*56);
end

Img = mat2gray(A, [min(min(A)) max(max(A))]);

imshow(Img);
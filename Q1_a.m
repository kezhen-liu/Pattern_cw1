clear;

TRAIN_NUM = 468;
TEST_NUM = 52;
EIGVEC_NUM = 467;

load('face.mat');

trainSet = zeros(2576, TRAIN_NUM);
testSet = zeros(2576, TEST_NUM);

testSetIndex = 1;
trainSetIndex = 1;
for i=1:520
    res = rem(i,10);
    if res == 0
        testSet(:,testSetIndex)=X(:,i);
        testSetIndex = testSetIndex + 1;
    else
        trainSet(:,trainSetIndex)=X(:,i);
        trainSetIndex = trainSetIndex + 1;
    end
end
trainMean = mean(trainSet.').';

%To plot the mean face
temp=zeros(56,46);
for i=1:46
    temp(:,i)=trainMean(1+(i-1)*56:i*56);
end
meanFaceImg = mat2gray(temp, [0 256]);
imshow(meanFaceImg);


for i = 1:TRAIN_NUM
    trainSet(:,i) = trainSet(:,i)-trainMean;
end

tic
trainCov = (trainSet*(trainSet.'))./TRAIN_NUM;
[eig_vec, eig_val] = eig(trainCov);
toc

mEigVal = zeros(1,EIGVEC_NUM);
mEigVec = zeros(2576,EIGVEC_NUM);
mEigIndex = 1;

for i=1:2576
    if eig_val(i,i) > 1
        mEigVal(1,mEigIndex) = eig_val(i,i);
        mEigVec(:,mEigIndex) = eig_vec(:,i);
        mEigIndex = mEigIndex+1;
    end
end

%{
%To plot the entire eigenvalues
plotEigValues=zeros(2576,1);
for i=1:2576
    plotEigValues(i)=eig_val(i,i);
end
plot(1:2576,plotEigValues);
title('Eigenvalues of training set');
ylabel('Values');
xlabel('#Eigenvector');

% To show the eigenfaces
Img=zeros(56,46,EIGVEC_NUM);
A=zeros(56,46);
for j=1:EIGVEC_NUM
    for i=1:46
        A(:,i)=mEigVec(1+(i-1)*56:i*56,j);
    end

    Img(:,:,j) = mat2gray(A, [min(min(A)) max(max(A))]);
end
imshow(Img(:,:,467));
%}
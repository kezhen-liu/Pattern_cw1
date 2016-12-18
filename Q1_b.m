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

%save('Q1_b_DataSet.mat', 'trainSet','testSet');

trainMean = mean(trainSet.').';
for i = 1:TRAIN_NUM
    trainSet(:,i) = trainSet(:,i)-trainMean;
end

tic
S = ((trainSet.')*trainSet)./TRAIN_NUM;
[V,D] = eig(S);
mEigVec = trainSet * V(:,TRAIN_NUM-EIGVEC_NUM+1:TRAIN_NUM);
toc

mD=zeros(TRAIN_NUM,1);

for i=1:TRAIN_NUM
    mD(i) = D(i,i);
end

for i=1:EIGVEC_NUM
    mEigVec(:,i) = mEigVec(:,i)/norm(mEigVec(:,i));
end
% To show the eigenfaces
Img=zeros(56,46,EIGVEC_NUM);
A=zeros(56,46);
for j=1:EIGVEC_NUM
    for i=1:46
        A(:,i)=mEigVec(1+(i-1)*56:i*56,j);
    end

    Img(:,:,j) = mat2gray(A, [min(min(A)) max(max(A))]);
end
imshow(Img(:,:,EIGVEC_NUM));

%mEigVec=fliplr(mEigVec);
%save('Q1_b_EigVec.mat', 'mEigVec');

clear;

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

trainCov = cov((trainSet).');
[eig_vec, eig_val] = eig(trainCov);

mEigVal = zeros(1,363);
mEigVec = zeros(2576,363);
mEigIndex = 1;

for i=1:2576
    if eig_val(i,i) > 100
        mEigVal(1,mEigIndex) = eig_val(i,i);
        mEigVec(:,mEigIndex) = eig_vec(:,i);
        mEigIndex = mEigIndex+1;
    end
end

% To show the eigenfaces
meanFace = mean(trainSet.').';
for i = 1:363
    mEigVec(:,i)= mEigVec(:,i) + meanFace;
end

Img=zeros(56,46,363);
A=zeros(56,46);
for j=1:363
    for i=1:46
        A(:,i)=mEigVec(1+(i-1)*56:i*56,j);
    end

    Img(:,:,j) = mat2gray(A, [0 255]);
end
imshow(Img(:,:,363));

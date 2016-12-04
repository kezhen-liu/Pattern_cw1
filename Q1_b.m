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

%S=cov(trainSet);
trainMean = mean(trainSet.').';
for i = 1:364
    trainSet(:,i) = trainSet(:,i)-trainMean;
end
S = ((trainSet.')*trainSet)./364;
[V,D] = eig(S);

mD=zeros(364,1);

for i=1:364
    mD(i) = D(i,i);
end

%mD=sort(mD);

mEigVec = trainSet * V;
mEigVec = mEigVec/norm(mEigVec);
% To show the eigenfaces
Img=zeros(56,46,363);
A=zeros(56,46);
for j=1:364
    for i=1:46
        A(:,i)=mEigVec(1+(i-1)*56:i*56,j);
    end

    Img(:,:,j) = mat2gray(A, [min(min(A)) max(max(A))]);
end
imshow(Img(:,:,364));

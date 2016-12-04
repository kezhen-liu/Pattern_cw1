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
    mD(i) = D(i,i)*364;
end

%mD=sort(mD);

% To show the eigenfaces
%{
meanFace = mean(trainSet).';
for i = 1:364
    V(:,i)= V(:,i) + meanFace;
end

Img=zeros(56,46,363);
A=zeros(56,46);
for j=1:363
    for i=1:46
        A(:,i)=V(1+(i-1)*56:i*56,j);
    end

    Img(:,:,j) = mat2gray(A, [0 255]);
end
imshow(Img(:,:,363));
     %}   

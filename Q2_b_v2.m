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

for i = 1:baseNum
    principleEigvec(:,i) = mEigVec(:,i);
end

tic
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

d2=zeros(int32(TRAIN_NUM/52),1);
d1=zeros(52,1);
diff=zeros(TEST_NUM,1);
outputCl=zeros(TEST_NUM,1);
for i=1:TEST_NUM % for each test face
    for j=1:52 % for each target class
        for k=1:int32(TRAIN_NUM/52) % for each face in a single target class
            d2(k)=norm(ax(:,(j-1)*int32(TRAIN_NUM/52)+k)-ay(:,i));
        end
        %{
        if i==4 && j == 4;
            disp(' ');
        end
        %}
        d1(j)=min(d2);
    end
    [diff(i),outputCl(i)]=min(d1);
end
toc

% Construct target class
targetCl=zeros(TEST_NUM,1);
for i=1:52
    for j=1:int32(TEST_NUM/52)
        targetCl((i-1)*int32(TEST_NUM/52)+j)=i;
    end
end

confusion=confusionmat(targetCl,outputCl);
imagesc(confusion);
colormap cool;
title('Confusion Matrix');
xlabel('Output Class');
ylabel('Target Class');

% To calculate accuracy
match=0;
for i=1:TEST_NUM
    if targetCl(i)==outputCl(i)
        match=match+1;
    end
end
accuray=match/TEST_NUM

%{
% Plot some specific faces
load('face.mat');
wrong=zeros(56,46);
test=zeros(56,46);
for i=1:46
    wrong(:,i)=X(1+(i-1)*56:i*56,38);
    test(:,i)=X(1+(i-1)*56:i*56,40);
end
I(:,:) = mat2gray(test, [0 256]);
subplot(1,2,1);
imshow(I(:,:));
title('Test face');
I(:,:) = mat2gray(wrong, [0 256]);
subplot(1,2,2);
imshow(I(:,:));
title('..best matches to..');
%}
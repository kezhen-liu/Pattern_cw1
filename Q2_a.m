clear;

TRAIN_NUM = 468;
TEST_NUM = 52;
EIGVEC_NUM = 467;

load('Q1_b_EigVec.mat');
load('Q1_b_DataSet.mat');


trainMean = mean(trainSet.').';
trainSetDiff = zeros(2576,TRAIN_NUM);
for i = 1:TRAIN_NUM
    trainSetDiff(:,i) = trainSet(:,i)-trainMean;
end

baseNumIndex = 1;
reconTrainImg = zeros(2576,4,3); %(pixcel, different number of bases, img)
reconTrain = zeros(EIGVEC_NUM,4,3); %(bases, different number of bases, img)

for baseNum=[10 50 100 EIGVEC_NUM] 
    imgIndex=1;
    for i=[1 1+int32(TRAIN_NUM/52) 1+2*int32(TRAIN_NUM/52)] % img number in train set
        reconTrain(1:baseNum,baseNumIndex,imgIndex) = (trainSetDiff(:,i).' * mEigVec(:,EIGVEC_NUM-baseNum+1:EIGVEC_NUM)).';
        imgIndex = imgIndex+1;
    end

    for i=1:3
        %for j=1:baseNum %change 364 here
        reconTrainImg(:,baseNumIndex,i) = (reconTrain(1:baseNum,baseNumIndex,i).' * mEigVec(:,EIGVEC_NUM-baseNum+1:EIGVEC_NUM).').';
        %end
        reconTrainImg(:,baseNumIndex,i) = reconTrainImg(:,baseNumIndex,i)+trainMean;
    end
    baseNumIndex = baseNumIndex + 1;
end

% To show the eigenfaces
IMG_TO_PLOT=1; %change 1 here
Img=zeros(56,46,4);
A=zeros(56,46);
for j=1:4
    for i=1:46
        A(:,i)=reconTrainImg(1+(i-1)*56:i*56,j,IMG_TO_PLOT);
    end
    Img(:,:,j) = mat2gray(A, [min(min(A)) max(max(A))]);
    subplot(1,5,j);
    imshow(Img(:,:,j));
    hold on;
end
subplot(1,5,5);
ref = zeros(56,46);
for i=1:46
        ref(:,i)=trainSet(1+(i-1)*56:i*56,(IMG_TO_PLOT-1)*int32(TRAIN_NUM/52)+1); 
end
Img1 = mat2gray(ref, [min(min(ref)) max(max(ref))]);
imshow(Img1);


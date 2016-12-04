clear;

load('Q1_b_EigVec.mat');
load('Q1_b_DataSet.mat');


trainMean = mean(trainSet.').';
trainSetDiff = zeros(2576,364);
for i = 1:364
    trainSetDiff(:,i) = trainSet(:,i)-trainMean;
end

baseNumIndex = 1;
reconTrainImg = zeros(2576,4,3); %(pixcel, different number of bases, img)
reconTrain = zeros(364,4,3); %(bases, different number of bases, img)

for baseNum=[10 50 100 364] 
    for i=[1 11 21] % img number in train set
        reconTrain(1:baseNum,baseNumIndex,int64(i/10+1)) = (trainSetDiff(:,i).' * mEigVec(:,1:baseNum)).';
    end

    for i=1:3
        for j=1:baseNum %change 364 here
            reconTrainImg(:,baseNumIndex,i) = reconTrainImg(:,baseNumIndex,i) + reconTrain(j,baseNumIndex,i) .* mEigVec(:,j);
        end
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
    subplot(1,4,j);
    imshow(Img(:,:,j));
    hold on;
end

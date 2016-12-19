clear;

TRAIN_NUM = 468;
TEST_NUM = 52;

config
OPTION_STR='-t 1 -q -g 1 -r 0 -d 2';

load('Q1_b_DataSet.mat');

trainSet=(trainSet-128)./128;

%train_scale_inst=sparse(((trainSet.')-128)./128);
train_scale_lable=double(-1.*ones(2*int32(size(trainSet,2)/52),1));
train_scale_lable(1:int32(size(trainSet,2)/52))=ones(int32(size(trainSet,2)/52),1);

tic
%Train each model
modelIndex=52*51/2;
modelLabel2faceID=int32(zeros(2,modelIndex));  %+1,-1
for i=51:-1:1
    for j=i+1:52 
        train_scale_inst=sparse(trainSet(:,[(i-1)*int32(size(trainSet,2)/52)+1:i*int32(size(trainSet,2)/52) (j-1)*int32(size(trainSet,2)/52)+1:j*int32(size(trainSet,2)/52)]).');
        model(modelIndex)=svmtrain(train_scale_lable, train_scale_inst, OPTION_STR);
        modelLabel2faceID(:,modelIndex)=[i,j];
        modelIndex=modelIndex-1;
    end
end
toc

disp('Start testing...');

tic
test_scale_inst=sparse(((testSet.')-128)./128);
predict_label=zeros(size(testSet,2),52*51/2); %test_face#, model#
accuracy=zeros(3,52*51/2); %[accuracy,mean square error,squared correlation coeff],model#
dec_values=zeros(size(testSet,2),52*51/2);    %test_face#,model#
%Test the entire test set by each model
for i=1:52*51/2    %for each model
    test_scale_lable=double(-1.*ones(size(test_scale_inst,1),1));   %we don't use predictor label here
    %test_scale_lable((i-1)*size(test_scale_inst,1)/52+1:i*size(test_scale_inst,1)/52)=ones(size(test_scale_inst,1)/52,1);
    [predict_label(:,i), accuracy(:,i), dec_values(:,i)] = svmpredict(test_scale_lable, test_scale_inst, model(i),'-q');
end

overall_predict=ones(size(testSet,2),1);
for i=1:size(testSet,2) %for each test face, count the vote
    vote=zeros(52,1);
    for j=1:52*51/2 %for each model, go through its decision value and make vote
        if dec_values(i,j)>=0
            vote(modelLabel2faceID(1,j))=vote(modelLabel2faceID(1,j))+1;
        else
            vote(modelLabel2faceID(2,j))=vote(modelLabel2faceID(2,j))+1;
        end
    end
    [M,indexMaxVote]=max(vote);
    overall_predict(i)=indexMaxVote;
end
toc

% Construct reference class label
targetCl=zeros(TEST_NUM,1);
for i=1:52
    for j=1:int32(TEST_NUM/52)
        targetCl((i-1)*int32(TEST_NUM/52)+j)=i;
    end
end

% Plot confusion matrix
confusion=confusionmat(targetCl,overall_predict);
imagesc(confusion);
colormap cool;
title('Confusion Matrix');
xlabel('Output Class');
ylabel('Target Class');

% Calculate on accuracy
match=0;
for i=1:TEST_NUM
    if overall_predict(i)==targetCl(i)
        match=match+1;
    end
end
accuracy=match/TEST_NUM


% To calculate margin
margin=zeros(52,2);
for i=1:52
    margin(i,1)=1/norm((model(i).sv_coef(1:model(i).nSV(1)).')*model(i).SVs(1:model(i).nSV(1),:));
    margin(i,2)=1/norm((model(i).sv_coef(model(i).nSV(1):model(i).totalSV).')*model(i).SVs(model(i).nSV(1):model(i).totalSV,:));
end
minMargin=min(min(margin))

%{
% Plot some specific faces
load('face.mat');
wrong=zeros(56,46);
test=zeros(56,46);
svPlot=zeros(56,46);
for i=1:46
    wrong(:,i)=X(1+(i-1)*56:i*56,379); %342
    test(:,i)=X(1+(i-1)*56:i*56,380); %340
    svPlot(:,i)=model(38).SVs(6,1+(i-1)*56:i*56).';
end
I(:,:) = mat2gray(test, [0 256]);
subplot(1,3,1);
imshow(I(:,:));
title('Test face');
I(:,:) = mat2gray(wrong, [0 256]);
subplot(1,3,2);
imshow(I(:,:));
title('..best matches to..');
subplot(1,3,3);
I(:,:) = mat2gray(svPlot, [-1 1]);
imshow(I(:,:));
title({'Most Sifnificant SV', ' of output class'});
%}
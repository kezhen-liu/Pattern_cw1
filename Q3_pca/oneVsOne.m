clear;
config
OPTION_STR='-t 0';

load('Q3_PCA_coeff.mat');

rangeTrainEigVal=max(max(trainEigVal))-min(min(trainEigVal));
rangeTestEigVal=max(max(testEigVal))-min(min(testEigVal));
range=max([rangeTrainEigVal rangeTestEigVal]);
trainEigVal=(trainEigVal-range/2)./(range/2);

%train_scale_inst=sparse(((trainSet.')-128)./128);
train_scale_lable=double(-1.*ones(2*int32(size(trainEigVal,2)/52),1));
train_scale_lable(1:int32(size(trainEigVal,2)/52))=ones(int32(size(trainEigVal,2)/52),1);

%Train each model
modelIndex=52*51/2;
modelLabel2faceID=int32(zeros(2,modelIndex));  %+1,-1
for i=51:-1:1
    for j=i+1:52 
        train_scale_inst=sparse(trainEigVal(:,[(i-1)*int32(size(trainEigVal,2)/52)+...
            1:i*int32(size(trainEigVal,2)/52) ...
            (j-1)*int32(size(trainEigVal,2)/52)+1:j*int32(size(trainEigVal,2)/52)]).');
        model(modelIndex)=svmtrain(train_scale_lable, train_scale_inst, OPTION_STR);
        modelLabel2faceID(:,modelIndex)=[i,j];
        modelIndex=modelIndex-1;
    end
end

disp('Start testing...');

test_scale_inst=sparse(((testEigVal.')-range/2)./(range/2));
predict_label=zeros(size(testEigVal,2),52*51/2); %test_face#, model#
accuracy=zeros(3,52*51/2); %[accuracy,mean square error,squared correlation coeff],model#
dec_values=zeros(size(testEigVal,2),52*51/2);    %test_face#,model#
%Test the entire test set by each model
for i=1:52*51/2    %for each model
    test_scale_lable=double(-1.*ones(size(test_scale_inst,1),1));   %we don't use predictor label here
    %test_scale_lable((i-1)*size(test_scale_inst,1)/52+1:i*size(test_scale_inst,1)/52)=ones(size(test_scale_inst,1)/52,1);
    [predict_label(:,i), accuracy(:,i), dec_values(:,i)] = svmpredict(test_scale_lable, test_scale_inst, model(i));
end

overall_predict=ones(size(testEigVal,2),1);
for i=1:size(testEigVal,2) %for each test face, count the vote
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
clear;

TRAIN_NUM = 468;
TEST_NUM = 52;

config
OPTION_STR='-t 2 -q -c 32 -g 0.00048828125 -w1 9 -w-1 1';

load('Q1_b_DataSet.mat');

train_scale_inst=sparse(((trainSet.')-128)./128);

tic
%Train each model
for i=52:-1:1    %for each person
    train_scale_lable=double(-1.*ones(size(train_scale_inst,1),1));
    train_scale_lable((i-1)*size(train_scale_inst,1)/52+1:i*size(train_scale_inst,1)/52)=ones(size(train_scale_inst,1)/52,1);
    model(i)=svmtrain(train_scale_lable, train_scale_inst, OPTION_STR);
end
toc

disp('Start testing...');

tic
test_scale_inst=sparse(((testSet.')-128)./128);
predict_label=zeros(size(testSet,2),52); %test_face#, model#
accuracy=zeros(3,52); %[accuracy,mean square error,squared correlation coeff],model#
dec_values=zeros(size(testSet,2),52);    %test_face#,model#
%Test the entire test set by each model
for i=1:52    %for each person
    test_scale_lable=double(-1.*ones(size(test_scale_inst,1),1));
    test_scale_lable((i-1)*size(test_scale_inst,1)/52+1:i*size(test_scale_inst,1)/52)=ones(size(test_scale_inst,1)/52,1);
    [predict_label(:,i), accuracy(:,i), dec_values(:,i)] = svmpredict(test_scale_lable, test_scale_inst, model(i),'-q');
end

overall_predict=ones(size(dec_values,1),1);
for i=1:size(dec_values,1)  %for each test face
    localMaxDec=dec_values(i,1);
    for j=1:52  %for each model, go through the decision values of that face and find the max
        if dec_values(i,j)>localMaxDec
            localMaxDec=dec_values(i,j);
            overall_predict(i)=j;
        end
    end
end
toc

% Construct reference class label
targetCl=zeros(TEST_NUM,1);
for i=1:52
    for j=1:int32(TEST_NUM/52)
        targetCl((i-1)*int32(TEST_NUM/52)+j)=i;
    end
end

match=0;
for i=1:TEST_NUM
    if overall_predict(i)==targetCl(i)
        match=match+1;
    end
end
accuracy=match/TEST_NUM
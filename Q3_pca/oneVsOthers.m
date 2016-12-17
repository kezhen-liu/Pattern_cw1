clear;
config
OPTION_STR='-t 0';

load('Q3_PCA_coeff.mat');

rangeTrainEigVal=max(max(trainEigVal))-min(min(trainEigVal));
rangeTestEigVal=max(max(testEigVal))-min(min(testEigVal));
range=max([rangeTrainEigVal rangeTestEigVal]);
train_scale_inst=sparse(((trainEigVal.')-range/2)./(range/2));

%Train each model
for i=52:-1:1    %for each person
    train_scale_lable=double(-1.*ones(size(train_scale_inst,1),1));
    train_scale_lable((i-1)*size(train_scale_inst,1)/52+1:i*size(train_scale_inst,1)/52)=ones(size(train_scale_inst,1)/52,1);
    model(i)=svmtrain(train_scale_lable, train_scale_inst, OPTION_STR);
end

disp('Start testing...');

test_scale_inst=sparse(((testEigVal.')-range/2)./(range/2));
predict_label=zeros(size(testEigVal,2),52); %test_face#, model#
accuracy=zeros(3,52); %[accuracy,mean square error,squared correlation coeff],model#
dec_values=zeros(size(testEigVal,2),52);    %test_face#,model#
%Test the entire test set by each model
for i=1:52    %for each person
    test_scale_lable=double(-1.*ones(size(test_scale_inst,1),1));
    test_scale_lable((i-1)*size(test_scale_inst,1)/52+1:i*size(test_scale_inst,1)/52)=ones(size(test_scale_inst,1)/52,1);
    [predict_label(:,i), accuracy(:,i), dec_values(:,i)] = svmpredict(test_scale_lable, test_scale_inst, model(i));
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
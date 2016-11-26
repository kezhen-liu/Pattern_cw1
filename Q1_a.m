clear;

load('face.mat');

trainSet = zeros(2576, 364);
testSet = zeros(2576, 156);

for i=1:52
    for j=1:7
        trainSet(:,j+10*(i-1)) = X(:,j+10*(i-1));
    end
    for j=8:10
        testSet(:,j-8+10(i-1))= X(:,j+10*(i-1));
clear;

load('face.mat');
A=zeros(56,46);
%{
for i=1:56
    A(i,:)=X(1+(i-1)*46:i*46,1);
end
%}
%I(520);
I=zeros(56,46,520);
for j=1:520
    for i=1:46
        A(:,i)=X(1+(i-1)*56:i*56,j);
    end

    I(:,:,j) = mat2gray(A, [0 256]);
    %subplot(26,20,j)
    %imshow(I(:,:,j));
end

for j=101:200
    subplot(10,10,j-100)
    imshow(I(:,:,j));
end
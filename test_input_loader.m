% an example to call input_loader.m
% create a 64 x 64 x 3 x 5 complex matrix and save it
a = zeros(64,64,3,5);
data = complex(a, 1);
save 'testdata.mat' data;
% load the matrix
data = input_loader('testdata.mat');
myBlob = Blob(data);
myBlob.get_data();
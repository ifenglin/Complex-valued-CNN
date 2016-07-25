% clear

%% use this data set for foroward propagation
% for convolution, activation, or pooling layers
data_real = ones(16, 16, 6);
data_imag = ones(16, 16, 6);

%% for convolution, activation, and pooling layers
% uncommment one of the line to test
myLayer = convolution_layer(5, 6, 1, 1, 0.01);
% myLayer = activation_layer('ReLU');
% myLayer = pooling_layer('MAX', 3, 3);

%% uncomment this block to test affine layer
% myLayer = affine_layer('ReLU', 128, 128, 0.01);
% data_real = ones(128, 1);
% data_imag = ones(128, 1);
% end of block

%% uncomment this block to test classifier layer
% myLayer = classification_layer('Magnitude', 5, 1e-1, 0.1);
% labels = [0 0 1 0 0]; % set the true lable
% myLayer = myLayer.set_labels(labels);
% data_real = ones(5, 1);
% data_imag = ones(5, 1);
% end of block

%% forward propagation - don't change
data =  complex(data_real, data_imag);
myBlob = Blob(data);
[myLayer, res_forward] = myLayer.forward(myBlob);
disp(res_forward)

%% for classifier, uncomment this block to show the estimated label
% [myLayer, res_forward, est_label] = myLayer.forward(myBlob);
% disp(res_forward)
% disp(est_label)
% end of block

%% For convinience, here the results in forward-propagation are used as 
% the input in backward-propagation, i.e. the derivatives.
% This is possible because they are always of the same dimension.
% However it has no mathematical meaning.
myBlob = Blob(data, res_forward);

%% backward propagation - dont' change
[myLayer, res_backward] = myLayer.backward(myBlob);
disp(res_backward);

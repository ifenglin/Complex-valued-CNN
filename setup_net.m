%% This files creates and initilizes a complex-valued convolutional network.

%% learning rates
% init_learning_rate and dec_learning_rate are defined for  convolutional 
% layers and affine layers. Convolution layers are given even smaller 
% learning rate because convoluting and pooling increase their values 
% during backpropagation.
% The learning rate is multiplied by dec_learning_rate after each epoch.
init_learning_rate = 1e-6;
dec_learning_rate = 0.95;

%% setup layers
% The default has two convolutional layers (three component layers each)
% and two fully connected layer of size 128.
% and a five-way classifier (implemented as one affine layer + one
% classfier)
% the inputs are default to a dimension 16 x 16 x 6
% The first convolutional layer 
% has 32 kernels of depth 6. The second convolutional layer 
% has 32 kernels of depth 6. 
conv_layer1 = convolution_layer(3, 6, 32, 1, init_learning_rate*1e-4);
acti_layer1 = activation_layer();
pool_layer1 = pooling_layer('MAX', 3, 3);
conv_layer2 = convolution_layer(3, 32, 128, 1, init_learning_rate*1e-2);
acti_layer2 = activation_layer();
pool_layer2 = pooling_layer('MAX', 3, 3);
aff_layer1 = affine_layer(128, 128, init_learning_rate);
aff_layer2 = affine_layer(128, 128, init_learning_rate);
aff_fiveway = affine_layer(128, 5, init_learning_rate);
classifer = classification_layer('Magnitude', 5, 0.005, 0);
layer_vec = [conv_layer1 acti_layer1 pool_layer1 conv_layer2 acti_layer2 pool_layer2 aff_layer1 aff_layer2 aff_fiveway classifer];
layer_names = {'conv-1', 'act-1' ,'pool-1', 'conv-2', 'act-2' ,'pool-2', 'fc1', 'fc2', 'fiveway', 'classifier'};
%% setup blobs automatically - don't change here
blob_names = cell(length(layer_vec), 1);
for i=1:length(layer_vec)
    blob_names{i} =  sprintf('Blob%d',i);
end
%% setup net
myNet = Net(layer_vec, layer_names, blob_names, [1], [length(layer_vec)]);
clear layer_vec layer_names blob_names
clear conv_layer1 acti_layer1 pool_layer1 conv_layer2 acti_layer2 pool_layer2 
clear aff_layer1 aff_layer2 aff_fiveway classifer
%% The default has two convolutional laters (three component layers each)
% and two fully connected layer of size 128.
% and a five-way classifier (implemented as one affine layer + one
% classfier)
% the inputs are default to a dimension 16 x 16 x 6

%% setup layers
% learning_rate is defined for affine layers
% convolution layers need even smaller learning rate owing to the
% convolution in backward-propagation.
learning_rate = 1e-4;

%obsoleted layers for images in 64 x 64
%conv_layer2 = convolution_layer(5, 6, 64, 1, 1e-8);
%acti_layer2 = activation_layer('ReLU');
%pool_layer2 = pooling_layer('MAX', 3, 3);

conv_layer3 = convolution_layer(3, 6, 32, 1, learning_rate*1e-6);
acti_layer3 = activation_layer('ReLU'); 
pool_layer3 = pooling_layer('MAX', 3, 3);
conv_layer4 = convolution_layer(3, 32, 128, 1, learning_rate*1e-3);
acti_layer4 = activation_layer('ReLU');
pool_layer4 = pooling_layer('MAX', 3, 3);
aff_layer1 = affine_layer('ReLU', 128, 128, learning_rate);
aff_layer2 = affine_layer('ReLU', 128, 128, learning_rate);
svm = affine_layer('SVM', 128, 5, learning_rate);
class_layer = classification_layer('Magnitude', 5, 1e-1, 0);

%layer_vec = [conv_layer2 acti_layer2 pool_layer2 conv_layer3 acti_layer3 pool_layer3 conv_layer4 acti_layer4 pool_layer4 aff_layer1 aff_layer2 svm class_layer];
%layer_names = {'conv2', 'activation2' ,'pooling2', 'conv3', 'activation3' ,'pooling3', 'conv4', 'activation4' ,'pooling4', 'fc1', 'fc2', 'svm', 'classifier'};
layer_vec = [conv_layer3 acti_layer3 pool_layer3 conv_layer4 acti_layer4 pool_layer4 aff_layer1 aff_layer2 svm class_layer];
layer_names = {'conv-1', 'act-1' ,'pool-1', 'conv-2', 'act-2' ,'pool-2', 'fc1', 'fc2', 'svm', 'classifier'};


%% setup blobs automatically - don't change here
blob_names = cell(length(layer_vec)+1, 1);
for i=1:length(layer_vec)+1
    blob_vec(i) = Blob();
    blob_names{i} =  sprintf('Blob%d',i);
end

%% setup net
myNet = Net(layer_vec, blob_vec, layer_names, blob_names, [1], [length(blob_vec)]);
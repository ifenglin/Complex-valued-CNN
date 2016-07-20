% clear;
% setup layers
conv_layer2 = convolution_layer(5, 3, 64, 1, 0.01);
acti_layer2 = activation_layer('ReLU');
pool_layer2 = pooling_layer('MAX', 3, 3);
conv_layer3 = convolution_layer(5, 64, 96, 1, 0.01);
acti_layer3 = activation_layer('ReLU');
pool_layer3 = pooling_layer('MAX', 3, 3);
conv_layer4 = convolution_layer(3, 96, 128, 1, 0.01);
acti_layer4 = activation_layer('ReLU');
pool_layer4 = pooling_layer('MAX', 3, 3);
aff_layer1 = affine_layer('ReLU', 128, 128, 0.01);
aff_layer2 = affine_layer('ReLU', 128, 128, 0.01);
svm = affine_layer('SVM', 128, 5, 0.01);
class_layer = classification_layer('Magnitude', 5, 1e-3, 1e-3);
layer_vec = [conv_layer2 acti_layer2 pool_layer2 conv_layer3 acti_layer3 pool_layer3 conv_layer4 acti_layer4 pool_layer4 aff_layer1 aff_layer2 svm class_layer];
layer_names = {'conv2', 'activation2' ,'pooling2', 'conv3', 'activation3' ,'pooling3', 'conv4', 'activation4' ,'pooling4', 'fc1', 'fc2', 'svm', 'classifier'};

% setup blobs automatically - don't change here
blob_names = cell(length(layer_vec)+1, 1);
for i=1:length(layer_vec)+1
    blob_vec(i) = Blob();
    blob_names{i} =  sprintf('Blob%d',i);
end

% setup net
myNet = Net(layer_vec, blob_vec, layer_names, blob_names, [1], [length(blob_vec)]);
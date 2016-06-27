clear;
% setup layers
conv_layer = convolution_layer(3, 3, 1);
acti_layer = activation_layer('ReLU');
pool_layer = pooling_layer('MAX', 3, 3);
layer_vec = [conv_layer acti_layer pool_layer];
layer_names = {'conv', 'pooling'};
% create test data
a = ones(64,64,3,1);
data = complex(a, 1);
% setup blobs
inputBlob = Blob(data);
blob_vec = [inputBlob Blob() Blob() Blob()];
blob_names = {'inputBlob', 'midBlob1', 'midBlob2', 'outputBlob'};
% setup net
myNet = Net(layer_vec, blob_vec, layer_names, blob_names, [1], [length(blob_vec)]);
inputs = {data};
% run
result = myNet.forward(inputs);
% print results
result(1).get_data()
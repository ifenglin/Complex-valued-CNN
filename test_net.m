clear;
% setup layers
conv_layer = convolution_layer(32, 5, 3, 1);
acti_layer = activation_layer('ReLU');
pool_layer = pooling_layer('MAX', 5, 5);
layer_vec = [conv_layer acti_layer pool_layer];
layer_names = {'conv', 'ReLU' ,'pooling'};
% create test data
data = rand(16,16,3,1);
data = complex(data, 1);
% setup blobs
blob_vec = [Blob() Blob() Blob() Blob()];
blob_names = {'inputBlob', 'midBlob1', 'midBlob2', 'outputBlob'};


% setup net
myNet = Net(layer_vec, blob_vec, layer_names, blob_names, [1], [length(blob_vec)]);
inputs = {data};
% run
[myNet, result_forward]  = myNet.forward(inputs);
a = result_forward.get_data(); % get the diff results of the first blob
% create test data
diff = rand(size(result_forward.get_data()));
diff = complex(diff, 1);
diffs = {diff};
[myNet, result_backward] = myNet.backward(diffs);
b = result_backward.get_diff(); % get the diff results of the first blob
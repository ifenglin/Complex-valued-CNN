clear;
% setup layers
conv_layer = convolution_layer(32, 5, 3, 1);
acti_layer = activation_layer('ReLU');
pool_layer = pooling_layer('MAX', 5, 5);
layer_vec = [conv_layer acti_layer pool_layer];
layer_names = {'conv', 'ReLU' ,'pooling'};
% create test data
data_real = rand(16,16,3,1);
data_imag = rand(16,16,3,1);
data = complex(data_real, data_imag);
% setup blobs
blob_vec = [Blob() Blob() Blob() Blob()];
blob_names = {'inputBlob', 'midBlob1', 'midBlob2', 'outputBlob'};


% setup net
myNet = Net(layer_vec, blob_vec, layer_names, blob_names, [1], [length(blob_vec)]);
inputs = {data};
% run forward-propagation
[myNet, result_forward]  = myNet.forward(inputs);
% get the diff results of the first blob
a = result_forward.get_data(); 

% create backward-propagation test data
diff_real = rand(size(result_forward.get_data()));
diff_imag = rand(size(result_forward.get_data()));
diff = complex(diff_real, diff_imag);
diffs = {diff};
% run backward-propagation
[myNet, result_backward] = myNet.backward(diffs);
% get the diff results of the first blob
b = result_backward.get_diff();
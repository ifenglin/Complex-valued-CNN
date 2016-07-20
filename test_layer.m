clear
% uncommment one of the layers to test
%myLayer = convolution_layer(3, 5, 3, 1, 0.01);
%myLayer = activation_layer('ReLU');
%myLayer = pooling_layer('MAX', 3, 3);
%myLayer = affine_layer('ReLU', 12, 3, 0.01);
myLayer = classification_layer('magnitude', 5, 0.1, 0.1);
%data_real = ones(12,12,3,1);
%data_imag = ones(12,12,3,1)*2;
%data = complex(data_real, data_imag);
data_real = ones(5,1);
data_imag = ones(5,1)*2;
data = complex(data_real, data_imag);
myBlob = Blob(data, data);
labels = [0 0 1 0 0];
[myLayer, label, res_forward] = myLayer.forward(myBlob, labels);
disp(label)
disp(res_forward)

data_real = ones(5,1);
data_imag = ones(5,1)*2;
data = complex(data_real, data_imag);
myBlob = Blob(data, data);

[myLayer, res_backward] = myLayer.backward(myBlob, labels);
disp(res_backward)
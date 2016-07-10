clear
% uncommment one of the layers to test
%myLayer = convolution_layer(3, 5, 3, 1);
%myLayer = activation_layer('ReLU');
myLayer = pooling_layer('MAX', 3, 3);
data_real = ones(12,12,3,1);
data_imag = ones(12,12,3,1)*2;
data = complex(data_real, data_imag);
myBlob = Blob(data, data);
[myLayer, res_forward] = myLayer.forward(myBlob);
disp(res_forward)
[myLayer, res_backward] = myLayer.backward(myBlob);
disp(res_backward)
% clear
% uncommment one of the layers to test
myLayer = convolution_layer(5, 1, 1, 1, 0.01);
%myLayer = activation_layer('ReLU');
%myLayer = pooling_layer('MAX', 3, 3);
%myLayer = affine_layer('ReLU', 12, 3, 0.01);
%myLayer = classification_layer('Magnitude', 5, 1e-1, 0.1);
%data_real = ones(12,12,3,1);
%data_imag = ones(12,12,3,1)*2;
%data = complex(data_real, data_imag);
data_real = ones(16, 16);
data_imag = ones(16, 16);
data = complex(data_real, data_imag);
myBlob = Blob(data);
labels = [0 0 1 0 0];

% forward propagation

[myLayer, res_forward] = myLayer.forward(myBlob);
disp(res_forward)

%[myLayer, label, res_forward] = myLayer.forward(myBlob, labels);
%disp(label)
%disp(res_forward)

% backward propagation
data_real = ones(12,12,1);
data_imag = ones(12,12,1);
data = complex(data_real, data_imag);
myBlob = Blob(data, data);

[myLayer, res_backward] = myLayer.backward(myBlob);
disp(res_backward);

%[myLayer, res_backward] = myLayer.backward(myBlob, labels);
%disp(res_backward)
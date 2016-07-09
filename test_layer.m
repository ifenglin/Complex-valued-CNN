clear
% uncommment one of the layers to test
%myLayer = convolution_layer(3, 5, 3, 1);
%myLayer = activation_layer('ReLU');
myLayer = pooling_layer('MAX', 3, 3);
data1 = ones(12,12,3,1);
data2 = ones(12,12,3,1)*2;
data = complex(data1, data2);
myBlob = Blob(data, data);
[myLayer, res1] = myLayer.forward(myBlob);
res1
[myLayer, res2] = myLayer.backward(myBlob);
res2
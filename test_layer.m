clear
% uncommment one of the layers to test
%myLayer = convolution_layer(3, 3, 1);
myLayer = activation_layer();
%myLayer = pooling_layer('MAX', 3, 3);
a = ones(64,64,3,1)*2;
data = complex(a, 1);
myBlob = Blob(data);
output_data = myLayer.forward(myBlob);
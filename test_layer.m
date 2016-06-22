clear
% uncommment one of the layers to test
%myLayer = convolution_layer(3, 3, 1);
%myLayer = pooling_layer('MAX', 3, 3);
a = zeros(64,64,3,1);
data = complex(a, 1);
myBlob = Blob(data);
output_data = myLayer.forward(myBlob)
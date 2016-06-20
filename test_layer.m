myLayer = convolution_layer(5, 5, 1);
a = zeros(12,12,3,1);
data = complex(a, 1);
myBlob = Blob(data);
output_data = myLayer.forward(myBlob)
myLayer = convolution_layer(5, 5, 1);
a = zeros(12,12,3);
data = complex(a, 1);
myBlob = Blob(data);
outputBlob = Blob(data);
% output_data = myLayer.forward(myBlob)

myNet = Net([myLayer], [myBlob, outputBlob], {'myLayer'}, {'myBlob', 'outputBlob'}, [1], [2]);
inputs = {data};
myNet.forward(inputs)
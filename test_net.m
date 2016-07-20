test_data = sample_loader(all_data, forest, (11:20));
% imshow(test_data(:,:,:,1))
% create labels
inputs = {test_data};
% label in order: city, field, forest, grass, street
labels = [0 0 1 0 0];

% train
[est_labels, losses]  = myNet.test(inputs, labels);
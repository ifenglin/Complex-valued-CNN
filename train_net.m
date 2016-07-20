% use sample_loader
test_data = sample_loader(all_data, field, (1:10));
% imshow(test_data(:,:,:,1))
% create labels
inputs = {test_data};
% label in order: city, field, forest, grass, street
labels = [0 1 0 0 0];   

% train
[myNet, est_labels_train, losses_train]  = myNet.train(inputs, labels);


% below run propagation separately - only for testing 

% run forward-propagation
%[myNet, est_label, result_forward]  = myNet.forward(inputs, labels);
% get the diff results of the first blob
%loss = result_forward.get_data(); 

% create backward-propagation test data
% diff_real = rand(size(result_forward.get_data()));
% diff_imag = rand(size(result_forward.get_data()));
% diff = complex(diff_real, diff_imag);
% diffs = {diff};
% run backward-propagation
%[myNet, result_backward] = myNet.backward();
% get the diff results of the first blob
%gradient = result_backward.get_diff();
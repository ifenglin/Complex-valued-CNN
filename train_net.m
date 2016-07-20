% use sample_loader
city = sample_loader(cm_all_data, cm_city, randperm(length(cm_city),10));
field = sample_loader(cm_all_data, cm_field, randperm(length(cm_field),10));
forest = sample_loader(cm_all_data, cm_forest, randi(length(cm_forest),10));
grass = sample_loader(cm_all_data, cm_grass, randi(length(cm_grass),10));
street = sample_loader(cm_all_data, cm_street, randi(length(cm_street),10));
% imshow(test_data(:,:,:,1))
% create labels
inputs = [{city} {field} {forest} {grass} {street}];
% label in order: city, field, forest, grass, street
labels = [ [1 0 0 0 0]; ...
           [0 1 0 0 0]; ...
           [0 0 1 0 0]; ...
           [0 0 0 1 0]; ...
           [0 0 0 0 1]];

% train
est_labels_train = zeros(5,10);
losses_train = zeros(5,10);
for j = 1:10
    for i = 1:5
        inputs = [{city(:,:,:,j)} {field(:,:,:,j)} {forest(:,:,:,j)} {grass(:,:,:,j)} {street(:,:,:,j)}];
        [myNet, est_labels_train(i,j), losses_train(i,j)]  = myNet.train(inputs(i), labels(i,:));
    end
end


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
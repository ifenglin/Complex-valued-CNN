num = 10;
% use sample_loader
city = sample_loader(cm_all_data, cm_city, randperm(length(cm_city), num));
field = sample_loader(cm_all_data, cm_field, randperm(length(cm_field), num));
forest = sample_loader(cm_all_data, cm_forest, randi(length(cm_forest), num));
grass = sample_loader(cm_all_data, cm_grass, randi(length(cm_grass), num));
street = sample_loader(cm_all_data, cm_street, randi(length(cm_street), num));
% imshow(test_data(:,:,:,1))
% create labels
inputs = [{city} {field} {forest} {grass} {street}];
% label in order: city, field, forest, grass, street
labels = [ [1 0 0 0 0]; ...
           [0 1 0 0 0]; ...
           [0 0 1 0 0]; ...
           [0 0 0 1 0]; ...
           [0 0 0 0 1]];
labels = repmat(labels, num, 1);
data = zeros(64,64,6,5*num);
 for i = 1:num
    for j = 1:5
        data(:,:,:,(i-1)*5+j) = inputs{j}(:,:,:,i);
    end
end
       
% train
[myNet, est_labels_train, res] = myNet.train({data}, labels);
train_svm = res(:,1:5);
train_mag_svm = arrayfun(@norm,result_svm);
train_loss = res(:,6);

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
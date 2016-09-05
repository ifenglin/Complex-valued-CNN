%% run setup_net.m first
% set up your test here
epochs = 10;
num = 20;
test_num = 5;
test_num_labels = 1;
num_all_labels = 5;
size = 16;
label_array = [1 2 3 4 5];
label_array = [2];
%label_array = [2 3];

% don't change below
num_labels = length(label_array);
% use sample_loader
city   = sample_loader(cm_all_data, cm_city,   randperm(length(cm_city),   num*epochs), size);
field  = sample_loader(cm_all_data, cm_field,  randperm(length(cm_field),  num*epochs), size);
forest = sample_loader(cm_all_data, cm_forest, randperm(length(cm_forest), num*epochs), size);
grass  = sample_loader(cm_all_data, cm_grass,  randperm(length(cm_grass),  num*epochs), size);
street = sample_loader(cm_all_data, cm_street, randperm(length(cm_street), num*epochs), size);
% create labels
inputs = [{city} {field} {forest} {grass} {street}];
inputs = inputs(label_array);
% prepare lable array
labels = zeros(num_labels, num_all_labels);
for i = 1:num_labels
    labels(i, label_array(i)) = 1;
end
labels = repmat(labels, num, 1);
% initial variables
data = zeros(size, size, 6, num*num_labels);
train_errors = zeros(num, num_labels);
train_loss = zeros(num);
test_errors = zeros(epochs+1, test_num*test_num_labels, num_all_labels);
test_loss = zeros(epochs+1, test_num*test_num_labels);
test_est_labels = zeros(epochs+1, test_num*test_num_labels);
test_correctness_rate = zeros(epochs+1, 1);
test_output_data = zeros(epochs+1, test_num*test_num_labels, num_all_labels);
% test before training
[test_errors(1,:,:), test_loss(1,:), test_est_labels(1,:), ...
    test_correctness_rate(1), test_output_data(1, :, :)] = f_test_net(myNet, size, test_num);
test_first_errors = squeeze(test_errors(1,:,:));

for epoch = 1:epochs
    disp(sprintf('## Epoch %d out of %d ##\n', epoch, epochs));
    % fetch data
    for i = 1:num
        for j = 1:num_labels
            data(:,:,:,(i-1)*num_labels+j) = inputs{j}(:,:,:,i+(epochs-1)*num);
        end
    end       
    % train
    [myNet, train_est_labels, loss] = myNet.train({data}, labels);
    train_errors = loss;
    train_loss = min(loss, [], 2);
    % test after each mini-batch
    [test_errors(epoch+1,:,:), test_loss(epoch+1,:), test_est_labels(epoch+1, :),...
        test_correctness_rate(epoch+1), test_output_data(epoch+1,:,:)] = ...
        f_test_net(myNet, size, test_num);
end
test_last_errors = squeeze(test_errors(epochs+1,:,:));
disp(sprintf('## %d epochs finished. Auf wiedersehen. ##\n', epochs));

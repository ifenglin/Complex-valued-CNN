%% run setup_params.m and setup_net.m first
%% Prerequisite: run setup_data.m to set variables inputs_test and 
%% inputs_train, or uncomment the line below to load from a mat file
load('data\inputs_train.mat');

% prepare testing data - select a set of labels in testing data and 
% only a subset of them accroding to params
inputs_test = inputs_test(test_known_labels);  
test_num_labels = length(test_known_labels);
test_data = zeros(size_patch, size_patch, num_channels, test_size_batch);
for i = 1:test_num_reps
    for j = 1:test_num_labels
        test_data(:,:,:,(i-1)*test_num_labels+j) = inputs_test{j}(:,:,:,i);
    end
end
% selece a set of labels in training data
inputs_train = inputs_train(train_known_labels);

% prepare lable array
labels = zeros(num_labels, num_all_labels);
for i = 1:num_labels
    labels(i, train_known_labels(i)) = 1;
end
labels = repmat(labels, num_reps, 1);

% initial variables
data = zeros(size_patch, size_patch, num_channels, size_batch);
train_errors = zeros(num_reps, num_labels);
train_loss = zeros(num_reps);
test_errors = zeros(test_num, test_size_batch, num_all_labels);
test_loss = zeros(test_num, test_size_batch);
test_est_labels = zeros(test_num, test_size_batch);
test_correctness_rate = zeros(test_num, 1);
test_output_data = zeros(test_num, test_size_batch, num_all_labels);
tic
for epoch = 1:num_epochs
    fprintf('## Epoch %d out of %d ##\n', epoch, num_epochs);
    % test after each epoch
    if mod(epoch-1, test_cycle) == 0 % test 10 times in all epoches
        fprintf('testing...');
        i = ceil(epoch/test_cycle);
        [test_errors(i,:,:), test_loss(i,:), test_est_labels(i, :), ...
            test_correctness_rate(i), test_output_data(i,:,:)] = ...
            f_test_net(myNet, test_data, test_known_labels, test_num_reps);
    end
    % shuffle the inputs by randomize their index
    inputs_index = arrayfun(@(K) randperm(size_epoch), 1:num_labels, 'UniformOutput', 0);
    % fetch data
    fprintf('training');
    for batch = 1:num_batches
        fprintf('.');
        for i = 1:num_reps
            data_index = (i-1)*num_labels;
            for j = 1:num_labels
                data(:,:,:,data_index+j) = inputs_train{j}(:,:,:,inputs_index{j}(i+(batch-1)*num_reps));
            end
        end
        % train
        [myNet, train_est_labels, train_errors, train_loss] = myNet.train({data}, labels);
    end
    myNet = myNet.set_dropout();
    myNet = myNet.set_learning_rate(dec_learning_rate);
    fprintf('\n');
end
fprintf('final testing...\n');
% test after training
[test_errors(num_epochs+1,:,:), test_loss(num_epochs+1,:), test_est_labels(num_epochs+1,:), ...
    test_correctness_rate(num_epochs+1), test_output_data(num_epochs+1, :, :)] = f_test_net(myNet, test_data, test_known_labels, test_num_reps);
fprintf('## %d epochs finished. Auf wiedersehen. ##\n', num_epochs);
toc
clear data
save 'data\myNet.mat' 'myNet' -v7.3

% analysis
for i = 1:length(train_known_labels)
    plot_outputs(test_size_batch, test_output_data, test_names_labels, train_known_labels(i))
    plot_errors(num_epochs, test_errors, test_names_labels, train_known_labels(i));
end
plot_loss(num_epochs, test_loss, test_correctness_rate);
clear inputs_test inputs_train 
save 'data\workspace_after_training.mat' -v7.3
%% run setup_net.m first
% data input dimension: size * size * num_channels
size = 1;
num_channels = 96;

% set up your test below
% number of epochs
num_epochs = 100;

% number of batches in one epoch
num_batches = 10;
% number of repeatance of given labels in one batch in training set
num_reps = 500;

% number of repeatance of given labels in testing set
test_num_reps = 100;
test_num_reserve = 10000;

% labels in training set
label_array = [1 2 3 4 5];

% number of labels in traing set
num_labels = length(label_array);

% number of labels in testing set
test_num_labels = 5;

% number of all labels
num_all_labels = 5;

% calculate mini-batch size for trainging set
size_batch = num_reps * num_labels;

% calculate mini-batch size for testing set
test_size_batch = test_num_reps * test_num_labels;

% calculate epoch size
size_epoch = num_reps * num_batches;

% do test after the number of epochs
test_cycle = 1;

% number of tests will be done over epochs
test_num = ceil(num_epochs / test_cycle) + 1;

%% prepare testing data
%% run create_test_data.m first and make sure it generates enough inputs of
% the same size (check variable 'num' and 'size')
% run setup_net.m first
% Important: use create_test_data.m to set inputs_test first
test_inputs_test = load('inputs_very_simple_test.mat');
%test_inputs_test = load('inputs_test.mat');
test_known_labels = [ 1 2 3 4 5 ];
test_num_labels = length(test_known_labels);
test_inputs_test = test_inputs_test.inputs_test(test_known_labels);  
test_data = zeros(size,size,num_channels,test_size_batch);
for i = 1:test_num_reps
    for j = 1:test_num_labels
        test_data(:,:,:,(i-1)*test_num_labels+j) = test_inputs_test{j}(:,:,:,i);
    end
end

%% prepare training data
% use sample_loader
% city   = sample_loader(cm_all_data, cm_city,   randperm(length(cm_city)-test_num_reserve, size_epoch)+test_num_reserve, 4);
% field  = sample_loader(cm_all_data, cm_field,  randperm(length(cm_field)-test_num_reserve, size_epoch)+test_num_reserve, 4);
% forest = sample_loader(cm_all_data, cm_forest, randperm(length(cm_forest)-test_num_reserve, size_epoch)+test_num_reserve, 4);
% grass  = sample_loader(cm_all_data, cm_grass,  randperm(length(cm_grass)-test_num_reserve, size_epoch)+test_num_reserve, 4);
% street = sample_loader(cm_all_data, cm_street, randperm(length(cm_street)-test_num_reserve, size_epoch)+test_num_reserve, 4);
% use very simple data
[city, field, forest, grass, street] = very_simple_data_loader(size, num_channels, size_epoch);
% create labels
inputs = [{city} {field} {forest} {grass} {street}];
inputs = inputs(label_array);
% create index
% inputs_index = zeros(num_labels, size_epoch);

% prepare lable array
labels = zeros(num_labels, num_all_labels);
for i = 1:num_labels
    labels(i, label_array(i)) = 1;
end
% inputs_index = arrayfun(@(K) randperm(size_epoch), 1:num_labels, 'UniformOutput', 0);

labels = repmat(labels, num_reps, 1);
% initial variables
data = zeros(size, size, num_channels, size_batch);
train_errors = zeros(num_reps, num_labels);
train_loss = zeros(num_reps);
test_errors = zeros(test_num, test_size_batch, num_all_labels);
test_loss = zeros(test_num, test_size_batch);
test_est_labels = zeros(test_num, test_size_batch);
test_correctness_rate = zeros(test_num, 1);
test_output_data = zeros(test_num, test_size_batch, num_all_labels);
tic
%test_first_errors = squeeze(test_errors(1,:,:));
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
                data(:,:,:,data_index+j) = inputs{j}(:,:,:,inputs_index{j}(i+(batch-1)*num_reps));
            end
        end
        myNet = myNet.set_dropout();
        % train
        [myNet, train_est_labels, train_errors, train_loss] = myNet.train({data}, labels);
    end
    fprintf('\n');
end
fprintf('final testing...\n');
% test before training
[test_errors(num_epochs+1,:,:), test_loss(num_epochs+1,:), test_est_labels(num_epochs+1,:), ...
    test_correctness_rate(num_epochs+1), test_output_data(num_epochs+1, :, :)] = f_test_net(myNet, test_data, test_known_labels, test_num_reps);
%test_last_errors = squeeze(test_errors(num_epochs+1, :, :));
fprintf('## %d epochs finished. Auf wiedersehen. ##\n', num_epochs);
toc
% analysis
label_plot = label_array(3);
plot_errors;
plot_loss;
plot_outputs;
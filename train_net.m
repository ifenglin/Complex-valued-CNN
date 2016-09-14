%% run setup_net.m first
% data input dimension: size * size * num_channels
size = 1;
num_channels = 96;

% set up your test below
% number of epochs
num_epochs = 30;

% number of batches in one epoch
num_batches = 1;

% number of repeatance of given labels in one batch in training set
num_reps = 5;

% number of repeatance of given labels in testing set
test_num_reps = 20;

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


% use sample_loader
%city   = sample_loader(cm_all_data, cm_city,   randperm(length(cm_city),   num*epochs), size);
%field  = sample_loader(cm_all_data, cm_field,  randperm(length(cm_field),  num*epochs), size);
%forest = sample_loader(cm_all_data, cm_forest, randperm(length(cm_forest), num*epochs), size);
%grass  = sample_loader(cm_all_data, cm_grass,  randperm(length(cm_grass),  num*epochs), size);
%street = sample_loader(cm_all_data, cm_street, randperm(length(cm_street), num*epochs), size);
% use very simple data
% use sample_loader
[city, field, forest, grass, street] = very_simple_data_loader(size, num_channels, size_epoch);
% create labels
inputs = [{city} {field} {forest} {grass} {street}];
inputs = inputs(label_array);
% create index
inputs_index = zeros(num_labels, num_reps*num_batches);

% prepare lable array
labels = zeros(num_labels);
for i = 1:num_labels
    labels(i, label_array(i)) = 1;
end
inputs_index = arrayfun(@(K) randperm(size_epoch), 1:num_labels, 'UniformOutput', 0);

labels = repmat(labels, num_reps, 1);
% initial variables
data = zeros(size, size, num_channels, size_batch);
train_errors = zeros(num_reps, num_labels);
train_loss = zeros(num_reps);
test_errors = zeros(num_epochs+1, test_size_batch, num_all_labels);
test_loss = zeros(num_epochs+1, test_size_batch);
test_est_labels = zeros(num_epochs+1, test_size_batch);
test_correctness_rate = zeros(num_epochs+1, 1);
test_output_data = zeros(num_epochs+1, test_size_batch, num_all_labels);
tic
% test before training
[test_errors(1,:,:), test_loss(1,:), test_est_labels(1,:), ...
    test_correctness_rate(1), test_output_data(1, :, :)] = f_test_net(myNet, size, num_channels, test_num_reps);
test_first_errors = squeeze(test_errors(1,:,:));
for epoch = 1:num_epochs
    fprintf('## Epoch %d out of %d ##\n', epoch, num_epochs);
    % fetch data
    fprintf('training');
    for batch = 1:num_batches
        fprintf('.');
        for i = 1:num_reps
            data_index = (i-1)*num_labels;
            parfor j = 1:num_labels
                data(:,:,:,data_index+j) = inputs{label_array(j)}(:,:,:,inputs_index{j}(i+(batch-1)*num_reps)); %#ok<PFBNS>
            end
        end       
        myNet = myNet.set_dropout();
        % train
        [myNet, train_est_labels, train_errors, train_loss] = myNet.train({data}, labels);
    end
    % test after each epoch
    fprintf('testing\n');
    [test_errors(epoch+1,:,:), test_loss(epoch+1,:), test_est_labels(epoch+1, :), ...
        test_correctness_rate(epoch+1), test_output_data(epoch+1,:,:)] = ...
        f_test_net(myNet, size, num_channels, test_num_reps);
    % shuffle the inputs by randomize their index
    inputs_index = arrayfun(@(K) randperm(size_epoch), 1:num_labels, 'UniformOutput', 0);
end
test_last_errors = squeeze(test_errors(num_epochs+1, :, :));
fprintf('## %d epochs finished. Auf wiedersehen. ##\n', num_epochs);
toc
% analysis
plot_errors;
plot_loss;
plot_outputs;
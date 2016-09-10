%% run create_test_data.m first and make sure it generates enough inputs of
% the same size (check variable 'num' and 'size')
% run setup_net.m first

% number of tests for each label
test_num = 5;
size = 16;
ch = 6;
test_num_labels = 5;
num_all_labels = 5;
%use create_test_data.m before testing
test_errors = zeros(1, test_num*test_num_labels, num_all_labels);
test_loss = zeros(1, test_num*test_num_labels);
test_est_labels = zeros(1, test_num*test_num_labels);
test_correctness_rate = zeros(1, 1);
test_output_data = zeros(1, test_num*test_num_labels, num_all_labels);

[test_errors(1,:,:), test_loss(1,:), test_est_labels(1, :),...
        test_correctness_rate(1), test_output_data(1,:,:)] = ...
        f_test_net(myNet, size, ch, test_num);
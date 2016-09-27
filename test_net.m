%% This programs is meant to evaluate the trained network with entire 
%% testing set. 
%% Prerequisite: run setup_data.m to prepare inputs_test.mat.
%% If the testing set is big, it takes a while.
%% Note: the testing set is always loaded from a file to save memory.
load 'data\inputs_test.mat'
tic
big_test_num_reps = test_num_reserve;
test_data = zeros(size_patch, size_patch, num_channels, test_num_labels * big_test_num_reps);
fprintf('evaluating the network...\n');
for i = 1:big_test_num_reps
    for j = 1:test_num_labels
        test_data(:,:,:,(i-1)*test_num_labels+j) = inputs_test{j}(:,:,:,i);
    end
end
[big_test_errors, big_test_loss, big_test_est_labels, ...
    big_test_correctness_rate, big_test_output_data] = f_test_net(myNet, test_data, test_known_labels, big_test_num_reps);
fprintf('The overall correctness rate is %f\n', big_test_correctness_rate);
for i=1:length(test_known_labels)
    fprintf('Label %s is correctly labeled at rate %f\n', test_names_labels{i}, sum(big_test_est_labels(i:test_num_labels:end) == test_known_labels(i))/big_test_num_reps);
end
toc
clear test_data
%% Warning: this program clears out variables inputs_test and inputs_train
%% to secure sufficient memory
%% This program label the data from all data with the network
%% Prerequisite: Run setup_net.m to set up a network and train_net.m to train it.
clear inputs_train inputs_test
load 'data\cm_alldata.mat'
% the number of pixels to skip at each step i.e. the resolution of the labeled image
stride_print = 100;
fprintf('labeling...\n');
tic
half_size = ceil(size_patch/2);
x = half_size:stride_print:size(cm_all_data, 1)-half_size-1;
y = half_size:stride_print:size(cm_all_data, 2)-half_size-1;
test_num = length(x) * length(y);
test_known_labels = 1; % this is dummy
test_data = new_sample_loader(cm_all_data, y, x);
        
[~, ~, test_image_est_labels, ~, test_image_output_data] = ...
    f_test_net(myNet, test_data, test_known_labels, test_num);

test_image_est_labels = reshape(test_image_est_labels, length(x), length(y));
toc
clear test_data

%% visualize
figure;
result_size  = size(test_image_est_labels);
result_label = zeros(result_size(1),result_size(2),3);

result_label(:,:,1) = test_image_est_labels==1;
result_label(:,:,1) = result_label(:,:,1) + (test_image_est_labels==2);
result_label(:,:,2) = (test_image_est_labels==3)*0.5;
result_label(:,:,2) = result_label(:,:,2) + (test_image_est_labels==2);
result_label(:,:,2) = result_label(:,:,2) + (test_image_est_labels==4);
result_label(:,:,3) = test_image_est_labels==5;

imshow(result_label);
clear result_size result_label result_label
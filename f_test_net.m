function [errors, loss, est_labels, correctness_rate, output_data] = f_test_net(myNet, test_data, test_known_labels, num)
    %% run create_test_data.m first and make sure it generates enough inputs of
    % the same size (check variable 'num' and 'size')
    % run setup_net.m first
    % Important: use create_test_data.m to set inputs_test first
    %inputs_test = load('inputs_very_simple_test.mat');
%     test_inputs_test = load('inputs_test.mat');
%     test_known_labels = [ 1 2 3 4 5 ];
%     test_num_labels = length(test_known_labels);
%     test_inputs_test = test_inputs_test.inputs_test(test_known_labels);  
%     test_data = zeros(size,size,num_channels,num);
%     for i = 1:num
%         for j = 1:test_num_labels
%             test_data(:,:,:,(i-1)*test_num_labels+j) = test_inputs_test{j}(:,:,:,i);
%         end
%     end
    % test
    test_num_all_labels = 5;
    test_num_labels = length(test_known_labels);
    test_known_labels = repmat(test_known_labels, 1, num)';
    [est_labels, errors, loss, output_data]  = myNet.test({test_data}, test_num_all_labels);
    correctness = sum(est_labels == test_known_labels);
    correctness_rate = squeeze(correctness / (num*test_num_labels));
end


function [errors, loss, est_labels, correctness_rate, output_data] = f_test_net(myNet, size, num_channels, num)
    %% run create_test_data.m first and make sure it generates enough inputs of
    % the same size (check variable 'num' and 'size')
    % run setup_net.m first
    % Important: use create_test_data.m to set inputs_test first
    %inputs_test = load('inputs_test.mat');
    inputs_test = load('inputs_very_simple_test.mat');
    % create labels
    % label in order: city, field, forest, grass, street
    true_labels = [ 1 2 3 4 5 ];
    num_all_labels = 5;
    num_labels = length(true_labels);
    inputs_test = inputs_test.inputs_test(true_labels);
    true_labels = repmat(true_labels, 1, num)';
    data = zeros(size,size,num_channels,num);
     for i = 1:num
        for j = 1:num_labels
            data(:,:,:,(i-1)*num_labels+j) = inputs_test{j}(:,:,:,i);
        end
    end
    
    % test
    [est_labels, errors, loss, output_data]  = myNet.test({data}, num_all_labels);
    correctness = sum(est_labels == true_labels);
    correctness_rate = squeeze(correctness / (num*num_labels));
end


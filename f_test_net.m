function [errors, loss, est_labels, correctness_rate, output_data] = f_test_net(myNet, test_data, test_known_labels, num)
    test_num_all_labels = 5;
    test_num_labels = length(test_known_labels);
    test_known_labels = repmat(test_known_labels, 1, num)';
    [est_labels, errors, loss, output_data]  = myNet.test({test_data}, test_num_all_labels);
    correctness = sum(est_labels == test_known_labels);
    correctness_rate = squeeze(correctness / (num*test_num_labels));
end


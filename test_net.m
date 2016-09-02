%% run create_test_data.m first and make sure it generates enough inputs of
% the same size (check variable 'num' and 'size')
% run setup_net.m first

% number of tests for each label
num = 10;
size = 16;
%use create_test_data.m before testing

% create labels
% label in order: city, field, forest, grass, street
labels = [ [1 0 0 0 0]; ...
           [0 1 0 0 0]; ...
           [0 0 1 0 0]; ...
           [0 0 0 1 0]; ...
           [0 0 0 0 1]]; 
labels = repmat(labels, num, 1);
data = zeros(size,size,6,num);
 for i = 1:num
    for j = 1:5
        data(:,:,:,(i-1)*5+j) = inputs_test{j}(:,:,:,i);
    end
end
[~, true_labels] = max(labels,[],2);
% test
[test_est_labels, loss]  = myNet.test({data}, labels);
test_errors = loss;
test_loss = min(loss, [], 2);
test_correctness = test_est_labels == true_labels;
test_correctness_rate = sum(test_correctness) / (num*5);

%% another style of testing
% can be useful in mini-batch-training
%for j = 1:num
%    for i = 1:5
%        inputs = [{city(:,:,:,j)} {field(:,:,:,j)} {forest(:,:,:,j)} {grass(:,:,:,j)} {street(:,:,:,j)}];
%        [est_labels_test(i,j), losses_test(i,j)]  = myNet.test(inputs(i), labels(i,:));
%    end
%end

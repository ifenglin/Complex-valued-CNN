% number of tests for each label
num = 2;
% use create_test_data.m before testing

% create labels
% label in order: city, field, forest, grass, street
labels = [ [1 0 0 0 0]; ...
           [0 1 0 0 0]; ...
           [0 0 1 0 0]; ...
           [0 0 0 1 0]; ...
           [0 0 0 0 1]]; 
labels = repmat(labels, num, 1);
data = zeros(64,64,6,num);
 for i = 1:num
    for j = 1:5
        data(:,:,:,(i-1)*5+j) = inputs_test{j}(:,:,:,i);
    end
end
[~, true_labels] = max(labels,[],2);
% test
[result_est_labels_test, res]  = myNet.test({data}, labels);
result_svm = res(:,1:5);
result_mag_svm = arrayfun(@norm,result_svm);
result_loss = res(:,6);
result_correctness = result_est_labels_test == true_labels;
result_correctness_rate = sum(result_correctness) / (num*5);

%for j = 1:num
%    for i = 1:5
%        inputs = [{city(:,:,:,j)} {field(:,:,:,j)} {forest(:,:,:,j)} {grass(:,:,:,j)} {street(:,:,:,j)}];
%        [est_labels_test(i,j), losses_test(i,j)]  = myNet.test(inputs(i), labels(i,:));
%    end
%end

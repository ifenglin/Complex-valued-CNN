num = 10;
% use sample_loader
city = sample_loader(cm_all_data, cm_city, randperm(length(cm_city),num));
field = sample_loader(cm_all_data, cm_field, randperm(length(cm_field),num));
forest = sample_loader(cm_all_data, cm_forest, randi(length(cm_forest),num));
grass = sample_loader(cm_all_data, cm_grass, randi(length(cm_grass),num));
street = sample_loader(cm_all_data, cm_street, randi(length(cm_street),num));
% imshow(test_data(:,:,:,1))
% create labels
inputs = [{city} {field} {forest} {grass} {street}];
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
        data(:,:,:,(i-1)*5+j) = inputs{j}(:,:,:,i);
    end
end
       
% test
[est_labels_test, losses_test]  = myNet.test({data}, labels);

%for j = 1:num
%    for i = 1:5
%        inputs = [{city(:,:,:,j)} {field(:,:,:,j)} {forest(:,:,:,j)} {grass(:,:,:,j)} {street(:,:,:,j)}];
%        [est_labels_test(i,j), losses_test(i,j)]  = myNet.test(inputs(i), labels(i,:));
%    end
%end

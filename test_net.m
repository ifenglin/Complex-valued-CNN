target = sample_loader(cm_all_data, cm_city, randperm(length(cm_city),1));
% use sample_loader
city = sample_loader(cm_all_data, cm_city, randperm(length(cm_city),10));
field = sample_loader(cm_all_data, cm_field, randperm(length(cm_field),10));
forest = sample_loader(cm_all_data, cm_forest, randi(length(cm_forest),10));
grass = sample_loader(cm_all_data, cm_grass, randi(length(cm_grass),10));
street = sample_loader(cm_all_data, cm_street, randi(length(cm_street),10));
% imshow(test_data(:,:,:,1))
% create labels
inputs = [{city} {field} {forest} {grass} {street}];
% label in order: city, field, forest, grass, street
labels = [ [1 0 0 0 0]; ...
           [0 1 0 0 0]; ...
           [0 0 1 0 0]; ...
           [0 0 0 1 0]; ...
           [0 0 0 0 1]];

test
est_labels_test = zeros(5,10);
losses_test = zeros(5,10);
for j = 1:10
    for i = 1:5
        inputs = [{city(:,:,:,j)} {field(:,:,:,j)} {forest(:,:,:,j)} {grass(:,:,:,j)} {street(:,:,:,j)}];
        [est_labels_test(i,j), losses_test(i,j)]  = myNet.test(inputs(i), labels(i,:));
    end
end

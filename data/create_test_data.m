%% this creates the input data for test_net.m
% run load_data.m first
num = 1000;
size = 16;

% use sample_loader
city = sample_loader(cm_all_data, cm_city, randperm(length(cm_city),num), size);
field = sample_loader(cm_all_data, cm_field, randperm(length(cm_field),num), size);
forest = sample_loader(cm_all_data, cm_forest, randi(length(cm_forest),num), size);
grass = sample_loader(cm_all_data, cm_grass, randi(length(cm_grass),num), size);
street = sample_loader(cm_all_data, cm_street, randi(length(cm_street),num), size);
% imshow(test_data(:,:,:,1))
inputs_test = [{city} {field} {forest} {grass} {street}];
save('inputs_test.mat', 'inputs_test');

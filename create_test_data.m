num = 10;
% use sample_loader
city = sample_loader(cm_all_data, cm_city, randperm(length(cm_city),num));
field = sample_loader(cm_all_data, cm_field, randperm(length(cm_field),num));
forest = sample_loader(cm_all_data, cm_forest, randi(length(cm_forest),num));
grass = sample_loader(cm_all_data, cm_grass, randi(length(cm_grass),num));
street = sample_loader(cm_all_data, cm_street, randi(length(cm_street),num));
% imshow(test_data(:,:,:,1))
inputs_test = [{city} {field} {forest} {grass} {street}];
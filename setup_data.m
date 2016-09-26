%% This program creates the inputs_train and inputs_test. 
%% They are necessary variables in order to run train_net.m 
%% Optionally, store them in file inputs.mat. It takes a while.
%% Prerequisite: run setup_params.m first 
save_inputs_train = 1;
load('data\cm_alldata.mat');
load('data\cm_labels.mat');

index_all_city = randperm(length(cm_city));
index_all_field = randperm(length(cm_field));
index_all_forest = randperm(length(cm_forest));
index_all_grass = randperm(length(cm_grass));
index_all_street = randperm(length(cm_street));
index_minitest_city = index_all_city(1:test_num_reps);
index_minitest_field = index_all_field(1:test_num_reps);
index_minitest_forest = index_all_forest(1:test_num_reps);
index_minitest_grass = index_all_grass(1:test_num_reps);
index_minitest_street = index_all_street(1:test_num_reps);
index_test_city = index_all_city(1:test_num_reserve);
index_test_field = index_all_field(1:test_num_reserve);
index_test_forest = index_all_forest(1:test_num_reserve);
index_test_grass = index_all_grass(1:test_num_reserve);
index_test_street = index_all_street(1:test_num_reserve);
index_train_city = index_all_city(test_num_reserve+1:end);
index_train_field = index_all_field(test_num_reserve+1:end);
index_train_forest = index_all_forest(test_num_reserve+1:end);
index_train_grass = index_all_grass(test_num_reserve+1:end);
index_train_street = index_all_street(test_num_reserve+1:end);

%% prepare big data
%% To save memory, this variabe will be cleared out after being stored
city = sample_loader(cm_all_data, cm_city, index_test_city, size_patch);
field = sample_loader(cm_all_data, cm_field, index_test_field, size_patch);
forest = sample_loader(cm_all_data, cm_forest, index_test_forest, size_patch);
grass = sample_loader(cm_all_data, cm_grass, index_test_grass, size_patch);
street = sample_loader(cm_all_data, cm_street, index_test_street, size_patch);
inputs_test = [{city} {field} {forest} {grass} {street}]; %#ok<NASGU>
save 'data\inputs_test.mat' 'inputs_test' -v7.3;
clear inputs_test

%% prepare minitest data
city = sample_loader(cm_all_data, cm_city, index_minitest_city, size_patch);
field = sample_loader(cm_all_data, cm_field, index_minitest_field, size_patch);
forest = sample_loader(cm_all_data, cm_forest, index_minitest_forest, size_patch);
grass = sample_loader(cm_all_data, cm_grass, index_minitest_grass, size_patch);
street = sample_loader(cm_all_data, cm_street, index_minitest_street, size_patch);
inputs_test = [{city} {field} {forest} {grass} {street}];

%% prepare training data
% use very simple data
%[city, field, forest, grass, street] = very_simple_data_loader(size, num_channels, size_epoch);
% use sample_loader
city   = sample_loader(cm_all_data, cm_city, index_train_city(1:size_epoch), size_patch);
field  = sample_loader(cm_all_data, cm_field, index_train_field(1:size_epoch), size_patch);
forest = sample_loader(cm_all_data, cm_forest, index_train_forest(1:size_epoch), size_patch);
grass  = sample_loader(cm_all_data, cm_grass, index_train_grass(1:size_epoch), size_patch);
street = sample_loader(cm_all_data, cm_street, index_train_street(1:size_epoch), size_patch);
% create labels
inputs_train = [{city} {field} {forest} {grass} {street}];
if (save_inputs_train == 1)
    save 'data\inputs_train.mat' 'inputs_test' -v7.3 'inputs_train' -v7.3
end
clear city field forest grass street
clear cm_all_data cm_city cm_field cm_forest cm_grass cm_street
clear index_all_city index_all_field index_all_forest index_all_grass index_all_street
clear index_test_city index_test_field index_test_forest index_test_grass index_test_street
clear index_minitest_city index_minitest_field index_minitest_forest index_mini_grass index_minitest_street
clear index_train_city index_train_field index_train_forest index_train_grass index_train_street

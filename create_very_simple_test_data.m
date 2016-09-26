%% this creates the input data for test_net.m
% run load_data.m first
num = 100;
size = 16;
ch = 6;
% create very simple data 
[city, field, forest, grass, street] = very_simple_data_loader(size, ch, num);

inputs_test = [{city} {field} {forest} {grass} {street}];
save('inputs_very_simple_test.mat', 'inputs_test');

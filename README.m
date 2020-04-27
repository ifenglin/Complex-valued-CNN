%% This is an instructional script for Matlab to set up a complex-valued neural 
%% network and, to train it with a dataset of 5 classes in mini-batch 
%% training, and finally to evalute the trained network with test and 
%% to visualize results.
%% For more information about this project, please refer to report.pdf

% The programs are developed in Matlab R2016a and Parallel Computing
% Toolbox Older versions may have conflicts in syntax.
% Note: in case no Parallel Computing Toolbox available, replace all the
% 'parfor' statement with 'for' to disable it.

%% Warning: we clear everything first.
clear

%% quick check?
% To test your environment, set this option to run the scripts with minimal
% data and loops.
quick_check = 1;

%% First, set up a network by runing setup_net.m
% By default, it has two convolutional layers (three component layers
% each) and two fully connected layer of size 128, and a five-way
% classifier. The network is designed for input dimension 16 x 16 x 6
% In the script there is also learning rate and its rate of change for
% tuning.
setup_net

%% Second, set up the parameters for training and testing
% In the first part, the dimension of the inputs are defined. The second
% part is training configuration, in which the number of epochs, batches,
% and inputs of each class are defined.  
% The third part, testing configuration, defines the inputs of used for
% testing during training. After every epoch in training, a subset of
% testing set is used to evaluate the intermediate perforance of the
% network.
% Note: consider the size of dataset when changing the parameters.
setup_params

%% Third, prepare training set and testing set.
% By default, it loads the variance-covariance matrix of PolSAR at 
% 'data\cm_alldata.mat' and its labels 'data\cm_labels.mat' and generate
% inputs_train.mat that has the training set and a subset of testing set,
% and inputs_test.mat that has the entire testing set.
% The number of each set is defined in setup_params.m
% Note: You might have to replace seperator '\' with '/' on non-Windows
% machines.
setup_data

%% Four, train the network.
% Learning curves over epochs are plotted at the end of training. In
% addition, for the purpose of analysis, it plots the average error of each
% class over selected tests as well as the actual outputs of the five-way 
% classifier.
train_net

%% Five, the entire testing set is used to evalute the trained network. 
% The correctness rate overall and of each class are printed in console.
% Note: You might have to replace seperator '\' with '/' on non-Windows
% machines.
test_net

%% Finally, label and print an image  
% By default, it loads 'data\cm_alldata.mat'
% The resolution of the image can be set in the file.
% Note: You might have to replace seperator '\' with '/' on non-Windows
% machines.
label_and_print
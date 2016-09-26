classdef Net
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties 
        layer_vec
        inputs              % names of blobs as input
        outputs             % names of blobs as output
        name2layer_index
        name2blob_index
        layer_names
        blob_names
    end
    methods
        function self = Net(layers, layerNames, blobNames, input_blob_indices, output_blob_indices)
            self.layer_vec = layers;
            self.layer_names = layerNames;
            self.blob_names = blobNames;
            
            self.inputs = self.blob_names(input_blob_indices);
            self.outputs = self.blob_names(output_blob_indices);
            self.name2layer_index = containers.Map(self.layer_names, 1:length(self.layer_names));
            self.name2blob_index = containers.Map(self.blob_names, 1:length(self.blob_names));
        end
        function layer = layers(self, layer_name)
            %CHECK(ischar(layer_name), 'layer_name must be a string');
            layer = self.layer_vec(self.name2layer_index(layer_name));
        end
        function [self, est_labels, errors, loss] = train(self, input_data, known_labels)
            % the 4th dimension is number of images
            num = size(input_data{:}, 4);
            est_labels = zeros(num, 1);
            errors = zeros(num, 5);
            loss = zeros(num, 1);
            delta_weights_cells = cell(num, 1);
            delta_bias_cells = cell(num, 1);
            parfor i = 1:num 
                %disp(sprintf('# Train image %d out of %d #\n', i, num));
                [my_blob_vec, ~, est_labels(i), errors(i,:), loss(i)] = self.forward({input_data{:}(:,:,:,i)}); %#ok<PFBNS>
                [~, delta_weights_cells{i}, delta_bias_cells{i}] = self.backward(my_blob_vec, known_labels(i,:));
            end
            mylayer = self.layer_vec;
            parfor i = 1:length(mylayer)
                sum_delta_weights{i} = zeros(size(mylayer(i).get_weights()));
                sum_delta_bias{i} = zeros(size(mylayer(i).get_bias()));
            end
            for i = 1:num
                for j = 1:length(mylayer)-1 % exclude classifier
                    sum_delta_weights{j} = sum_delta_weights{j} + delta_weights_cells{i}{j};
                    sum_delta_bias{j} = sum_delta_bias{j} + delta_bias_cells{i}{j};
                end
            end
            self = self.update(sum_delta_weights, sum_delta_bias);
%             for i = 1:num
%                 self = self.set_delta(delta_weights_cells{i}, delta_bias_cells{i});
%                 self = self.update(delta_weights_cells{i}, delta_bias_cells{i});
%             end
            %disp(sprintf('# Trained with %d inputs and updated.#\n', num));
        end
        
        function [est_labels, errors, loss, output_data] = test(self, input_data, num_labels)
            % the 4th dimension is number of images
            num = size(input_data{:}, 4);
            est_labels = zeros(num, 1);
            errors = zeros(num, num_labels);
            loss = zeros(num, 1);
            output_data = zeros(num, num_labels);
            
            parfor i = 1:num 
                %disp(sprintf('# Test image %d #\n', i));
                [~, output_data(i,:), est_labels(i), errors(i,:), loss(i)] = self.forward({input_data{:}(:,:,:,i)}); %#ok<PFBNS>
            end
        end
        
        function self = update(self, delta_weights_cells, delta_bias_cells)
            my_layer_vec = self.layer_vec;
            new_layer_vec = [];
            for i = 1:length(self.layer_vec)-1 % exclude classifier
                new_layer = my_layer_vec(i).update(delta_weights_cells{i}, delta_bias_cells{i});
                new_layer_vec = horzcat(new_layer_vec, new_layer);
            end
            new_layer_vec = horzcat(new_layer_vec, my_layer_vec(length(self.layer_vec)));
            self.layer_vec = new_layer_vec;
        end
        
        function [blob_vec, output_data, est_label, errors, loss] = forward(self, input_data)
            parfor i=1:length(self.layer_vec)
                blob_vec(i) = Blob();
            end
            % copy data to input blobs
            %disp('#Forward Propagation#');
            for n = 1:length(self.inputs)
                blob_vec(self.name2blob_index(self.inputs{n})) = ...
                    blob_vec(self.name2blob_index(self.inputs{n})).set_data(input_data{n});
            end
            % layer i takes blob i and store results in blob i+1
            % check the number of layers is exactly one fewer than
            % the number of blobs 
            assert(length(self.layer_vec) == length(blob_vec));
            
            % weight regularization: acummualte weight vector for classifier
%             weight_cells = [];
            for i = 1:length(self.layer_vec)
                %disp(char(self.layer_names(i)));
                if ~strcmp(self.layer_vec(i).get_type(), 'classification')
                    [~, my_output_data] = self.layer_vec(i).forward(blob_vec(i)); 
                    blob_vec(i+1) = blob_vec(i+1).set_data(my_output_data);
                    % weight regularization: store weight matrix
%                     weight_cells = [weight_cells; reshape(self.layer_vec(i).get_weights(), [], 1)]; 
                else % if classifier
                    % weight regularization: assign weight_cells to classifier
%                     self.layer_vec(end) = self.layer_vec(end).set_weight_cells(weight_cells);
                    % forward and display estimated label index
                    [~, output_data, est_label, errors, loss] = self.layer_vec(i).forward(blob_vec(i));
                end
            end
        end
        function [blob_vec, delta_weights_cells, delta_bias_cells] = backward(self, blob_vec, known_labels, input_diff)
            %disp('#Backward Propagation#');
            assert(length(self.layer_vec) == length(blob_vec));
            % copy diff to output blobs
            if nargin > 3
                for n = 1:length(self.outputs)
                    blob_vec(self.name2blob_index(self.outputs{n})) = ...
                        blob_vec(self.name2blob_index(self.outputs{n})).set_diff(input_diff{n});
                end
            end
            delta_weights_cells = cell(length(self.layer_vec)-1, 1);
            delta_bias_cells = cell(length(self.layer_vec)-1, 1);
            % layer i takes blob i+1 and store results in blob i
            % check the number of layers is exactly one fewer than
            % the number of blobs 
            for i = fliplr(1:length(self.layer_vec))
                if i == length(self.layer_vec) % classifier
                    [~, output_diff] = self.layer_vec(i).backward(blob_vec(i), known_labels);
                else
                    [~, output_diff, delta_weights_cells{i}, delta_bias_cells{i}] = self.layer_vec(i).backward(blob_vec(i));
                end
                if i ~= 1
                    blob_vec(i-1) = blob_vec(i-1).set_diff(output_diff);
                end
            end 
        end
%         function [delta_weights_cells, delta_bias_cells] = get_delta_cells(self)
%             mylayer_vec = self.layer_vec;
%             delta_weights_cells = cell(length(self.layer_vec), 1);
%             delta_bias_cells = cell(length(self.layer_vec), 1);
%             for i = 1:length(mylayer_vec)
%                 [delta_weights_cells{i}, delta_bias_cells{i}] = mylayer_vec(i).get_delta();
%             end
%         end
%         function self = set_delta(self, delta_weights_cells, delta_bias_cells)
%             for i = 1:length(self.layer_vec)-1
%                 self.layer_vec(i).set_delta(delta_weights_cells{i}, delta_bias_cells{i});
%             end
%         end
        function self = set_dropout(self)
            mylayer_vec = self.layer_vec;
            for i = 1:length(self.layer_vec)
               new_layer_vec(i) = mylayer_vec(i).set_dropout();
            end
            self.layer_vec = new_layer_vec;
        end
        function self = set_learning_rate(self, alpha_percetage)
            mylayer_vec = self.layer_vec;
            for i = 1:length(self.layer_vec)
               new_layer_vec(i) = mylayer_vec(i).set_learning_rate(alpha_percetage);
            end
            self.layer_vec = new_layer_vec;
        end
    end
end


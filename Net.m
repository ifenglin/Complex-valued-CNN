classdef Net
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties 
        layer_vec
        blob_vec
        inputs              % names of blobs as input
        outputs             % names of blobs as output
        name2layer_index
        name2blob_index
        layer_names
        blob_names
    end
    methods
        function self = Net(layers, blobs, layerNames, blobNames, input_blob_indices, output_blob_indices)
            self.layer_vec = layers;
            self.blob_vec = blobs;
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
        function blob = blobs(self, blob_name)
            %CHECK(ischar(blob_name), 'blob_name must be a string');
            assert (ischar(blob_name))
            blob = self.blob_vec(self.name2blob_index(blob_name));
        end
        function blob = params(self, layer_name, blob_index)
            %CHECK(ischar(layer_name), 'layer_name must be a string');
            %CHECK(isscalar(blob_index), 'blob_index must be a scalar');
            blob = self.layer_vec(self.name2layer_index(layer_name)).params(blob_index);
        end
        function forward_prefilled(~)
        end
        function backward_prefilled(~)
        end
        function [self, est_labels, loss] = train(self, input_data, labels)
            % the 4th dimension is number of images
            num = size(input_data{:}, 4);
            est_labels = zeros(num, 1);
            loss = zeros(num, 5);
            
            for i = 1:num 
                disp(sprintf('#### Train image %d out of %d ####\n', i, num));
                [self, est_labels(i), loss(i,:)] = self.forward({input_data{:}(:,:,:,i)} , labels(i,:));
                %losses(i,:) = res.get_data();
                self = self.backward();
            end
            self = self.update();
            disp(sprintf('#### The network is trained with %d inputs and weights are updated.####\n', num));
        end
        
        function [est_labels, loss] = test(self, input_data, labels)
            % the 4th dimension is number of images
            num = size(input_data{:}, 4);
            est_labels = zeros(num, 1);
            loss = zeros(num, size(labels, 2));
            for i = 1:num 
                disp(sprintf('#### Test image %d ####\n', i));
                [self, est_labels(i), loss(i,:)] = self.forward({input_data{:}(:,:,:,i)} , labels(i,:));
            end
        end
        
        function self = update(self)
            for i = 1:length(self.layer_vec)
                self.layer_vec(i) = self.layer_vec(i).update();
            end
        end
        
        
        function [self, est_label, loss] = forward(self, input_data, labels)
            % copy data to input blobs
            disp('#Forward Propagation#');
            for n = 1:length(self.inputs)
                self.blob_vec(self.name2blob_index(self.inputs{n})) = ...
                    self.blob_vec(self.name2blob_index(self.inputs{n})).set_data(input_data{n});
            end
            % self = self.forward_prefilled();
            % layer i takes blob i and store results in blob i+1
            % check the number of layers is exactly one fewer than
            % the number of blobs 
            assert(length(self.layer_vec) == length(self.blob_vec)-1);
            
            % acummualte weight vector for classifier
            weight_vector = [];
            for i = 1:length(self.layer_vec)
                disp(char(self.layer_names(i)));
                %tic
                if ~strcmp(self.layer_vec(i).get_type(), 'classification')
                    [self.layer_vec(i), data] = self.layer_vec(i).forward(self.blob_vec(i));
                    self.blob_vec(i+1) = self.blob_vec(i+1).set_data(data);
                    weight_vector = [weight_vector; reshape(self.layer_vec(i).get_weights(), [], 1)]; %#ok<AGROW>
                else % if classifier
                    % assign labels to classifier
                    self.layer_vec(i) = self.layer_vec(end).set_labels(labels);
                    % assign weight_vector to classifier
                    self.layer_vec(i) = self.layer_vec(end).set_weight_vector(weight_vector);
                    % forward and display estimated label index
                    [self.layer_vec(i), data, est_label, loss] = self.layer_vec(i).forward(self.blob_vec(i));
                    self.blob_vec(i+1) = self.blob_vec(i+1).set_data(data);
                end
                %toc
            end
            
            % retrieve data from output blobs
            %res(1,length(self.outputs)) = Blob();
            %for n = 1:length(self.outputs)
            %    res(n) = self.blobs(self.outputs{n});
            %end
        end
        function [self, res] = backward(self, input_diff)
            disp('#Backward Propagation#');
            % copy diff to output blobs
            if nargin > 1
                for n = 1:length(self.outputs)
                    self.blob_vec(self.name2blob_index(self.outputs{n})) = ...
                        self.blob_vec(self.name2blob_index(self.outputs{n})).set_diff(input_diff{n});
                end
            end
            
            % self.backward_prefilled();
            % layer i takes blob i+1 and store results in blob i
            % check the number of layers is exactly one fewer than
            % the number of blobs 
            assert(length(self.layer_vec) == length(self.blob_vec)-1);
            for i = fliplr(1:length(self.layer_vec))
                disp(char(self.layer_names(i)));
                %tic
                [self.layer_vec(i), diff] = self.layer_vec(i).backward(self.blob_vec(i+1));
                self.blob_vec(i) = self.blob_vec(i).set_diff(diff);
                %toc
            end 
            % retrieve diff from input blobs
            %res(1,length(self.outputs)) = Blob();
            %for n = 1:length(self.inputs)
            %  res(n) = self.blobs(self.inputs{n});
            %end
        end
        function copy_from(self, weights_file)
            CHECK(ischar(weights_file), 'weights_file must be a string');
            CHECK_FILE_EXIST(weights_file);
            % load to nets
        end
        function save(save, weights_file)
            CHECK(ischar(weights_file), 'weights_file must be a string');
        end
    end
end


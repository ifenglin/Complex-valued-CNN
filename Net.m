classdef Net
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        layer_vec
        blob_vec
        inputs % names of blobs as input
        outputs % names of blobs as output
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
            blob = self.blob_vec(self.name2blob_index(blob_name));
        end
        function blob = params(self, layer_name, blob_index)
            %CHECK(ischar(layer_name), 'layer_name must be a string');
            %CHECK(isscalar(blob_index), 'blob_index must be a scalar');
            blob = self.layer_vec(self.name2layer_index(layer_name)).params(blob_index);
        end
        function forward_prefilled(self)
            % do something
        end
        function backward_prefilled(self)
            % do something
        end
        function res = forward(self, input_data)
            %CHECK(iscell(input_data), 'input_data must be a cell array');
            %CHECK(length(input_data) == length(self.inputs), ...
            %  'input data cell length must match input blob number');
            % copy data to input blobs
            for n = 1:length(self.inputs)
                self.blobs(self.inputs{n}).set_data(input_data{n});
            end
            self.forward_prefilled();
            % retrieve data from output blobs
            res = cell(length(self.outputs), 1);
            for n = 1:length(self.outputs)
                res{n} = self.blobs(self.outputs{n}).get_data();
            end
        end
        function res = backward(self, output_diff)
            CHECK(iscell(output_diff), 'output_diff must be a cell array');
            CHECK(length(output_diff) == length(self.outputs), ...
              'output diff cell length must match output blob number');
            % copy diff to output blobs
            for n = 1:length(self.outputs)
              self.blobs(self.outputs{n}).set_diff(output_diff{n});
            end
            self.backward_prefilled();
            % retrieve diff from input blobs
            res = cell(length(self.inputs), 1);
            for n = 1:length(self.inputs)
              res{n} = self.blobs(self.inputs{n}).get_diff();
            end
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


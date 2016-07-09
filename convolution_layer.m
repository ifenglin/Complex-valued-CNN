classdef convolution_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        num_output  % number of filters
        kernel_size % filter size in pixels (squared)
        group      % number of groups/channels 
        stride      % number of pixels to step between each filter application
        kernels
    end
    
    methods
        function self = convolution_layer(num_output, kernel_size, group, stride)
            % setup self handle and attributes
            self@Layer('convolution');
            self.num_output = num_output;
            self.kernel_size = kernel_size;
            self.group = group;
            self.stride = stride;
            rand_kernels_real = rand(group, num_output);
            rand_kernels_imag = rand(group, num_output);
            self.kernels = complex(rand_kernels_real, rand_kernels_imag);
        end
        function layer_type = get_type(self)
            layer_type = self.type;
        end
        function [self, output_data] = forward(self, input_blob)
            num_channels = input_blob.get_num_channels();
            output_data = complex(ones( ...
                (input_blob.get_height() - self.kernel_size) / self.stride + 1, ...
                (input_blob.get_width() - self.kernel_size) / self.stride + 1, ...
                self.num_output));  % the size of output data is shrinked dimensions by num_output
            
            x = 1:self.stride:input_blob.get_height() - self.kernel_size + 1;
            y = 1:self.stride:input_blob.get_width() - self.kernel_size + 1;
            for i = 1:length(x)
                for j = 1:length(y)
                    patch_reshaped = input_blob.get_data(x(i):x(i)+self.kernel_size-1, y(j):y(j)+self.kernel_size-1);
                    patch_reshaped = reshape(patch_reshaped, [], num_channels); 
                    output_data(i, j, :) = sum(patch_reshaped * self.kernels, 1);      
                end
            end
            % output_blob = input_blob.set_data(output_data);
        end
        function [self, output_diff] = backward(self, input_blob)
            num_channels = input_blob.get_num_channels();
            output_diff = complex(ones( ...
                (input_blob.get_height() - 1) / self.stride + self.kernel_size, ...
                (input_blob.get_width() - 1) / self.stride + self.kernel_size, ...
                num_channels));  % the size of output data is the expended dimensions by num_output
            
            pad = complex(ones(input_blob.get_height() + self.kernel_size + 1, ...
                        input_blob.get_width() + self.kernel_size + 1, ...
                        self.num_output), 0);
            pad(floor(self.kernel_size-1)/2 : input_blob.get_height() + floor(self.kernel_size-1)/2 -1, ...
                floor(self.kernel_size-1)/2 : input_blob.get_width() +  floor(self.kernel_size-1)/2 -1, ...
                 :) = complex(input_blob.get_diff());
            
            x = 1:self.stride:input_blob.get_height()  + (self.stride * self.kernel_size) - 1;
            y = 1:self.stride:input_blob.get_width()  + (self.stride * self.kernel_size) - 1;
            for i = 1:length(x)
                for j = 1:length(y)
                    % sum a pad in x, y directions and sum all feature maps
                    output_diff(i,j,:) = sum(sum(sum(pad(x:x+self.kernel_size - 1, y:y+self.kernel_size - 1, :) ,1) ,2),3); 
                end
            end
            % output_blob = input_blob.set_diff(output_data);
        end
    end
    
end


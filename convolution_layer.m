classdef convolution_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        num_output  % number of filters
        kernel_size % filter size in pixels (squared)
        num_group      % number of groups/channels 
        stride      % number of pixels to step between each filter application
        kernels
        bias
        alpha      % learning rate
        forwarded_input_data  % keep a copy for weight update
    end
    
    methods
        function self = convolution_layer(kernel_size, group, num_output, stride, alpha)
            % setup self handle and attributes
            self@Layer('convolution');
            self.num_output = num_output;
            self.kernel_size = kernel_size;
            self.num_group = group;
            self.stride = stride;
            % Note : weights in kernels and bias shares the same parameters 
            % at different locations in the image.
            % initialize kernels 
            rand_kernels_real = rand(kernel_size, kernel_size, group, num_output ) - 0.5;
            rand_kernels_imag = rand(kernel_size, kernel_size, group, num_output ) - 0.5;
            self.kernels = complex(rand_kernels_real, rand_kernels_imag);
            % initialize bias
            rand_bias_real = zeros(num_output, 1);
            rand_bias_imag = zeros(num_output, 1);
            self.bias = complex(rand_bias_real, rand_bias_imag);
            self.alpha = alpha;
        end
        function kernels = get_weights(self)
            kernels = self.kernels;
        end
        function [self, output_data] = forward(self, input_blob)
            self.forwarded_input_data = input_blob.get_data();
            output_data = complex(ones( ...
                (input_blob.get_height() - self.kernel_size) / self.stride + 1, ...
                (input_blob.get_width() - self.kernel_size) / self.stride + 1, ...
                self.num_output));  % the size of output data is shrinked dimensions by num_output
            % index of the left top corner of a patch in pad
            x = 1:self.stride:input_blob.get_height() - self.kernel_size + 1;
            y = 1:self.stride:input_blob.get_width() - self.kernel_size + 1;
            for i = 1:length(x)
                for j = 1:length(y)
                    % get a patch at ( x(i), y(i) ) as the left-top corner
                    % with kernel_size^2
                    patch = input_blob.get_data(x(i):x(i)+self.kernel_size-1, y(j):y(j)+self.kernel_size-1);
                    
                    % replicate patch to all depth
                    patch_array = repmat(patch, 1, 1, 1, self.num_output);
                    
                    % multiple the patch with kernels in all depth
                    output_depth = squeeze( ...
                        sum(sum(patch_array .* self.kernels, 1 ),2 )...
                    );    
                    %output_depth is 1 by 1 by num_group by num_output
                    output_data(i, j, :) = sum(output_depth)' + self.bias;
                end
            end
            % output_blob = input_blob.set_data(output_data);
        end
        function [self, output_diff] = backward(self, input_blob)
            height = size(self.forwarded_input_data, 1);
            width = size(self.forwarded_input_data, 2);
            output_diff = complex(zeros(size(self.forwarded_input_data)));  % the size of output data is the expended dimensions by num_output
            % calculate the additional pixels needed for de-convolute on the
            % boundaries
            pad = complex(zeros(size(input_blob.get_diff())));
            %pad = complex( zeros(input_blob.get_height() + self.kernel_size + 1, ...
            %            input_blob.get_width() + self.kernel_size + 1, ...
            %            self.num_output), 0);
            pad(floor(self.kernel_size-1)/2 + 1: input_blob.get_height() + floor(self.kernel_size-1)/2, ...
                floor(self.kernel_size-1)/2 + 1: input_blob.get_width() +  floor(self.kernel_size-1)/2, ...
                 :) = complex(input_blob.get_diff());
            % index of the left top corner of a patch in pad
            % x = 1:self.stride:input_blob.get_height()  + (self.stride * self.kernel_size) - 1;
            % y = 1:self.stride:input_blob.get_width()  + (self.stride * self.kernel_size) - 1;
            x = 1:self.stride:height - self.kernel_size + 1;
            y = 1:self.stride:width - self.kernel_size + 1;
            for i = 1:length(x)
                for j = 1:length(y)
                    % update kernels
                    % copy the values into all channels
                    % For programming reasons, copy the channels in the
                    % forth dimension and then swap the third and forth
                    % dimension
                    pad_array = permute( ...
                        repmat(pad(x:x+self.kernel_size - 1, y:y+self.kernel_size - 1, :), 1, 1, 1, self.num_group),...
                        [1 2 4 3]);
                    forwarded_input_data_array = repmat(self.forwarded_input_data(x:x+self.kernel_size - 1, y:y+self.kernel_size - 1, :), 1, 1, 1, self.num_output);
                    self.kernels = self.kernels - self.alpha * forwarded_input_data_array .* pad_array;
                    
                    % update bias with the sum in a pad 
                    self.bias = self.bias - self.alpha * ...
                        reshape(sum(sum(pad(x:x+self.kernel_size - 1, y:y+self.kernel_size - 1, :) ) ),self.num_output, 1 );
                    
                    % sum all feature maps as output at (i, j)
                    output_diff(i,j,:) = sum(sum(sum(pad_array, 1), 2), 4 );
                end
            end
            
           
            % output_blob = input_blob.set_diff(output_data);
        end
    end
    
end


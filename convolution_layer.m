classdef convolution_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        num_output       % number of filters
        kernel_size      % filter size in pixels (squared)
        num_groups        % number of groups/channels 
        stride           % number of pixels to step between each filter application
        kernels          % weights
        bias            
        alpha            % learning rate
    end
    
    methods
        function self = convolution_layer(kernel_size, group, num_output, stride, alpha)
            % setup self handle and attributes
            self@Layer('convolution');
            self.num_output = num_output;
            self.kernel_size = kernel_size;
            self.num_groups = group;
            self.stride = stride;
            % Note : weights in kernels and bias shares the same parameters 
            % at different locations in the image.
            % initialize kernels 
            num_inputs_per_kernel = kernel_size * kernel_size;
            rand_kernels_real = normrnd(0, 1, kernel_size, kernel_size, group, num_output)*sqrt(2.0/(num_inputs_per_kernel*group));
            rand_kernels_imag = normrnd(0, 1, kernel_size, kernel_size, group, num_output)*sqrt(2.0/(num_inputs_per_kernel*group));
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
        function bias = get_bias(self)
            bias = self.bias;
        end
        function self = set_learning_rate(self, alpha_percetage)
            self.alpha = self.alpha * alpha_percetage;
        end
         function alpha = get_learning_rate(self)
            alpha = self.alpha;
        end
        function [self, output_data] = forward(self, input_blob)
            input_data = input_blob.get_data();
            output_data = complex(ones( ...
                (input_blob.get_height() - self.kernel_size) / self.stride + 1, ...
                (input_blob.get_width()  - self.kernel_size) / self.stride + 1, ...
                self.num_output));  % the size of output data is shrinked dimensions by num_output
            % index of the left top corner of a patch in pad
            x = 1:self.stride:input_blob.get_height() - self.kernel_size + 1;
            y = 1:self.stride:input_blob.get_width() - self.kernel_size + 1;
            my_kernel_size = self.kernel_size;
            my_num_output = self.num_output;
            my_kernels = self.kernels;
            my_bias = self.bias;
            for i = 1:length(x)
                data_x = input_data(x(i):x(i)+ my_kernel_size-1, :, :);
                for j = 1:length(y)
                    % get a patch at ( x(i), y(i) ) as the left-top corner
                    % with kernel_size^2
                    patch = data_x(:, y(j):y(j)+my_kernel_size-1, :);
                    
                    % replicate patch to all depth
                    patch_array = repmat(patch, [1, 1, 1, my_num_output]);
                    
                    % multiple the patch with kernels in all depth
                    output_depth = squeeze( ...
                        sum(sum(patch_array .* my_kernels, 1 ),2 )...
                    );    
                    %output_depth is 1 by 1 by num_groups by num_output
                    output_data(i, j, :) = reshape(sum(output_depth),my_num_output,1) + my_bias;
                end
            end
        end
        function [self, output_diff, kernels_delta, bias_delta] = backward(self, input_blob)
            input_data = input_blob.get_data();
            input_diff = input_blob.get_diff();
            height = size(input_data, 1);
            width = size(input_data, 2);
            % the size of output data is the expended dimensions by num_output
            output_diff = complex(zeros(size(input_data)));  
            % index of the left top corner of a patch in pad
            x = 1:self.stride:height - self.kernel_size + 1;
            y = 1:self.stride:width - self.kernel_size + 1;
            my_kernel_size = self.kernel_size;
            my_num_groups = self.num_groups;
            kernels_delta = zeros(size(self.kernels));
            bias_delta = zeros(size(self.bias));
            for i = 1:length(x)
                for j = 1:length(y)
                    % update kernels
                    % copy the values into all channels
                    % For programming reasons, copy the channels in the
                    % forth dimension and then swap the third and forth
                    % dimension
                    pad_array = permute( ...
                        repmat(input_diff(i, j, :), [my_kernel_size, my_kernel_size, 1, my_num_groups]), [1 2 4 3]);
                    % sum all feature maps as output at (i, j)
                    output_diff(x(i):x(i)+my_kernel_size - 1, y(j):y(j)+my_kernel_size - 1, :) = ...
                        output_diff(x(i):x(i)+my_kernel_size - 1, y(j):y(j)+my_kernel_size - 1, :) + ...
                        sum(pad_array, 4);
                    kernels_delta = kernels_delta + self.alpha * pad_array;
                    bias_delta = bias_delta + self.alpha * squeeze(sum(sum(sum(pad_array, 1), 2), 3));
                end
            end
        end
        
        function self = update(self, kernels_delta, bias_delta)
            self.kernels = self.kernels + kernels_delta;
            self.bias = self.bias + bias_delta;
        end
    end
    
end


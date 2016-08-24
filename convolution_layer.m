classdef convolution_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        num_output       % number of filters
        kernel_size      % filter size in pixels (squared)
        num_group        % number of groups/channels 
        stride           % number of pixels to step between each filter application
        kernels          % weights
        bias            
        alpha            % learning rate
        forwarded_input_data  % keep a copy for weight update
        kernels_delta    % updated weights to apply when updating
        bias_delta       % updated bias to apply when updating
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
            rand_kernels_real = rand(kernel_size, kernel_size, group, num_output)*2 - 1;
            rand_kernels_imag = rand(kernel_size, kernel_size, group, num_output)*2 - 1;
            self.kernels = complex(rand_kernels_real, rand_kernels_imag);
            % initialize bias
            rand_bias_real = ones(num_output, 1);
            rand_bias_imag = ones(num_output, 1);
            self.bias = complex(rand_bias_real, rand_bias_imag);
            self.alpha = alpha;
            self.kernels_delta = complex(zeros(size(self.kernels)));
            self.bias_delta = complex(zeros(size(self.bias)));
        end
        function kernels = get_weights(self)
            kernels = self.kernels;
        end
        function self = set_learning_rate(self, alpha)
            self.alpha = alpha;
        end
         function alpha = get_learning_rate(self)
            alpha = self.alpha;
        end
        function [self, output_data] = forward(self, input_blob)
            self.forwarded_input_data = input_blob.get_data();
            output_data = complex(ones( ...
                (input_blob.get_height() - self.kernel_size) / self.stride + 1, ...
                (input_blob.get_width()  - self.kernel_size) / self.stride + 1, ...
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
                    patch_array = repmat(patch, [1, 1, 1, self.num_output]);
                    
                    % multiple the patch with kernels in all depth
                    output_depth = squeeze( ...
                        sum(sum(patch_array .* self.kernels, 1 ),2 )...
                    );    
                    %output_depth is 1 by 1 by num_group by num_output
                    output_data(i, j, :) = reshape(sum(output_depth),self.num_output,1) + self.bias;
                end
            end
            % output_blob = input_blob.set_data(output_data);
        end
        function [self, output_diff] = backward(self, input_blob)
            height = size(self.forwarded_input_data, 1);
            width = size(self.forwarded_input_data, 2);
            height_diff = size(input_blob.get_diff(),1);
            width_diff = size(input_blob.get_diff(),2);
            % the size of output data is the expended dimensions by num_output
            output_diff = complex(zeros(size(self.forwarded_input_data)));  
            % calculate the additional pixels needed for de-convolute on the
            % boundaries
            pad = complex(zeros(height, width, self.num_output), 0);
            pad(floor(self.kernel_size-1)/2 + 1: height_diff + floor(self.kernel_size-1)/2, ...
                floor(self.kernel_size-1)/2 + 1: width_diff +  floor(self.kernel_size-1)/2, ...
                 :) = input_blob.get_diff();
            % index of the left top corner of a patch in pad
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
                        repmat(pad(x(i):x(i)+self.kernel_size - 1, y(i):y(i)+self.kernel_size - 1, :), [1, 1, 1, self.num_group]),...
                        [1 2 4 3]);
                    forwarded_input_data_array = repmat(self.forwarded_input_data(x:x+self.kernel_size - 1, y:y+self.kernel_size - 1, :), [1, 1, 1, self.num_output]);
                    
                    
                    kernels_new = self.kernels - self.alpha * forwarded_input_data_array .* pad_array;
                    % limit the value of weights in
                    % [sqrt(num_output), sqrt(num_output)]
                    % compare in real value domain
                    limit_kernels = ones(size(kernels_new))*sqrt(self.num_output);  
                    kernels_real = real(kernels_new);
                    kernels_imag = imag(kernels_new);
                    kernels_real = max(min(kernels_real, limit_kernels), -limit_kernels);
                    kernels_imag = max(min(kernels_imag, limit_kernels), -limit_kernels);
                    self.kernels_delta = self.kernels_delta + complex(kernels_real, kernels_imag);
                    
                    % update bias with the sum in a pad 
                    bias_new = self.bias - self.alpha * ...
                       reshape(sum(sum(pad(x:x+self.kernel_size - 1, y:y+self.kernel_size - 1, :) ) ),self.num_output, 1 );
                    % limit the value of bias in
                    % [sqrt(num_output), sqrt(num_output)]
                    % compare in real value domain
                    limit_bias = ones(size(bias_new))*sqrt(self.num_output);  
                    bias_real = real(bias_new);
                    bias_imag = imag(bias_new);
                    bias_real = max(min(bias_real, limit_bias), -limit_bias);
                    bias_imag = max(min(bias_imag, limit_bias), -limit_bias);
                    self.bias_delta = self.bias_delta + complex(bias_real, bias_imag);
                    
                    % sum all feature maps as output at (i, j)
                    output_diff(x(i):x(i)+self.kernel_size - 1, y(j):y(j)+self.kernel_size - 1, :) = ...
                        output_diff(x(i):x(i)+self.kernel_size - 1, y(j):y(j)+self.kernel_size - 1, :) + ...
                        sum(pad_array, 4);
                end
            end

            % output_blob = input_blob.set_diff(output_data);
        end
        
        function self = update(self)
            self.kernels = self.kernels + self.kernels_delta;
            self.bias = self.bias + self.bias_delta;
            self.kernels_delta = complex(zeros(size(self.kernels)));
            self.bias_delta = complex(zeros(size(self.bias))); 
        end
    end
    
end


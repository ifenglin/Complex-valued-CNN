classdef convolution_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        num_output  % number of filters
        kernel_size % filter size in pixels (squared)
        stride      % number of pixels to step between each filter application
    end
    
    methods
        function self = convolution_layer(num_output, kernel_size, stride)
            % setup self handle and attributes
            self@Layer('convolution');
            self.num_output = num_output;
            self.kernel_size = kernel_size;
            self.stride = stride;
        end
        function layer_type = get_type(self)
            layer_type = self.type;
        end
        function output_data = forward(self, input_data)
            %prepare kernel
            kernel = ones(self.kernel_size, self.kernel_size);
            kernel = complex(kernel);
            kernel = reshape(kernel, 1, []); % reshaped into one row
            output_data = ones( ...
                (input_data.get_height() - self.kernel_size) / self.stride + 1, ...
                (input_data.get_width() - self.kernel_size) / self.stride + 1);
            x = 1:self.stride:input_data.get_height() - self.kernel_size + 1;
            y = 1:self.stride:input_data.get_width() - self.kernel_size + 1;
            for i = 1:length(x)
                for j = 1:length(y)
                    onepatch = input_data.get_data(x(i):x(i)+self.kernel_size-1, y(j):y(j)+self.kernel_size-1);
                    onepatch = reshape(onepatch,[],1); % reshaped into one column 
                    output_data(i, j) = kernel*onepatch;
                    str = sprintf('(%d,%d) = ', i, j);
                    disp(str);
                    disp(output_data(i, j));
                end
            end
            % do something
        end
        function backward()
            % do something
        end
    end
    
end


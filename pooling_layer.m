classdef pooling_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        pool        % pooling method
        kernel_size % filter size in pixels (squared)
        stride      % number of pixels to step between each filter application
    end
    
    methods (Access = public)
        function self = pooling_layer(pool, kernel_size, stride)
            % setup self handle and attributes
            self@Layer('pooling');
            self.pool = pool;
            self.kernel_size = kernel_size;
            self.stride = stride;
        end
        function output_blob = forward(self, input_data)
            % prepare kernel
            %kernel = ones(self.kernel_size, self.kernel_size);
            %kernel = complex(kernel);
            %kernel = reshape(kernel, 1, []); % reshaped into one row
            % prepare pad - add extra pixels in the boundaries to avoid
            % incomplete pooling
            margin_x = self.stride - mod(input_data.get_height(),self.stride);
            margin_y = self.stride - mod(input_data.get_width(),self.stride);
            num_channels = input_data.get_channels();
            pad = ones(input_data.get_height() + margin_x,...
                       input_data.get_width() + margin_y,...
                       num_channels);
            pad(floor(margin_x/2)+1:end-ceil(margin_x/2), floor(margin_y/2)+1:end-ceil(margin_y/2), :) = input_data.get_data();
           
            output_data = ones( ...
                ( size(pad, 1) - self.kernel_size) / self.stride + 1, ...
                ( size(pad, 2) - self.kernel_size) / self.stride + 1, ...
                num_channels);
            x = 1:self.stride:input_data.get_height() - self.kernel_size + 1;
            y = 1:self.stride:input_data.get_width() - self.kernel_size + 1;
            
            for i = 1:length(x)
                for j = 1:length(y)
                    onepatch = pad(x(i):x(i)+self.kernel_size-1, y(j):y(j)+self.kernel_size-1,:);
                    onepatch = reshape(onepatch,[],num_channels);
                    output_data(i, j, :) = self.pooling(onepatch);
                    %str = sprintf('%d-(%d,%d) = ', k, i, j);
                    %disp(str);
                    %disp(output_data(i, j, k));
                end
            end
            output_blob = Blob(output_data);
        end
        function backward()
            % do something
        end
    end
    methods (Access = private)
        function output = pooling(self, patch_vector)
            switch self.pool
                case 'MAX'
                    output = max(patch_vector, [], 1);
                case 'SOFTMAX'
                    %TODO
            end
                
        end
    end 
end


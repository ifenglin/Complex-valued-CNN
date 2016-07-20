classdef pooling_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        pool        % pooling method
        kernel_size % filter size in pixels (squared)
        stride      % number of pixels to step between each filter application
        switches
        w_margin
        h_margin
        w_range
        h_range
    end
    
    methods (Access = public)
        function self = pooling_layer(pool, kernel_size, stride)
            assert(kernel_size == stride, 'Oops...kernel size and stride must be the same!')
            % setup self handle and attributes
            self@Layer('pooling');
            self.pool = pool;
            self.kernel_size = kernel_size;
            self.stride = stride;
        end
        function [self, output_data] = forward(self, input_blob)
            % calculate the additional pixels needed for pooling on the
            % boundaries
            self.h_margin = self.stride - mod(input_blob.get_height(),self.stride);
            self.w_margin = self.stride - mod(input_blob.get_width(),self.stride);
            % calculate the range of index of height and width of inputs in
            % an extended (margin-inclusive) pad
            self.h_range = ceil(self.h_margin/2) : input_blob.get_height()+ceil(self.h_margin/2)-1;
            self.w_range = ceil(self.w_margin/2) : input_blob.get_width()+ceil(self.w_margin/2)-1;
            num_channels = input_blob.get_num_channels();
            % create a pad and put the input data incide.
            pad = zeros(input_blob.get_height() + self.h_margin,...
                       input_blob.get_width() + self.w_margin,...
                       input_blob.get_num_channels());
            pad(self.h_range, self.w_range, :) = input_blob.get_data();
           
            output_data = ones( ...
                ( size(pad, 1) - self.kernel_size) / self.stride , ...
                ( size(pad, 2) - self.kernel_size) / self.stride , ...
                num_channels);
            % swtiches set 1 if the pixel in pad is activated; 0 otherwise
            temp_switches = zeros(size(pad));
            % index of the left top corner of a patch in pad
            x = 1:self.stride:size(pad,1) - self.kernel_size;
            y = 1:self.stride:size(pad,2) - self.kernel_size;
            
            for i = 1:length(x)
                for j = 1:length(y)
                    % get a patch of kernel_size^2 in pad
                    patch_x = x(i):x(i)+self.kernel_size-1;
                    patch_y = y(j):y(j)+self.kernel_size-1;
                    onepatch = pad(patch_x, patch_y,:);
                    onepatch_columns = reshape(onepatch,[],num_channels); % reshape one channel into one column
                    [output_data(i, j, :), index] = self.pooling(onepatch_columns);
                    switch_mask = zeros(self.kernel_size^2, num_channels); % the same size as onepatch_columns
                    for k = 1:length(index)
                        switch_mask(index(k), k) = 1; % set true on the activated position 
                    end 
                    temp_switches(patch_x, patch_y, :) = reshape(switch_mask, self.kernel_size, self.kernel_size, num_channels);
                end
            end
            self.switches = temp_switches(self.h_range, self.w_range, :);
            %output_blob = input_blob.set_data(output_data);
        end
        function [self, output_diff] = backward(self, input_blob)
            % calculate the additional pixels needed for pooling on the
            % boundaries
            pad = zeros(size(self.switches,1) + self.h_margin,...
                        size(self.switches,2) + self.w_margin,...
                        size(self.switches,3));
            pad(self.h_range, self.w_range, :) = self.switches;
            num_channels = input_blob.get_num_channels();
            % index of the left top corner of a patch in pad
            x = 1:self.stride:size(pad,1) - self.kernel_size;
            y = 1:self.stride:size(pad,2) - self.kernel_size;
            
            for i = 1:length(x)
                for j = 1:length(y)
                    % index range corresponding to x and y
                    patch_x = x(i):x(i)+self.kernel_size-1;
                    patch_y = y(j):y(j)+self.kernel_size-1;
                    % assign the diff to the activated cell, looping for
                    % every chanel
                    for k = 1:num_channels
                        pad(patch_x, patch_y,k) = pad(patch_x, patch_y,k) * input_blob.get_diff(i, j, k);
                    end
                    %onepatch_columns = onepatch_columns * reshape(input_blob.get_diff(i, j),[], num_channels)';
                    %pad(patch_x, patch_y,:) = reshape(onepatch_columns, self.kernel_size, self.kernel_size, num_channels);
                end
            end 
            output_diff = pad(self.h_range, self.w_range, :);
            %output_blob = input_blob.set_diff(output_data);
        end
    end
    
    methods (Access = private)
        function [output, index] = pooling(self, patch_vector)
            switch self.pool
                case 'MAX'
                    [output, index] = max(patch_vector, [], 1);
                case 'SOFTMAX'
                    %TODO
            end
        end
    end 
end


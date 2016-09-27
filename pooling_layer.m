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
            h_margin = self.stride - mod(input_blob.get_height(),self.stride);
            w_margin = self.stride - mod(input_blob.get_width(),self.stride);
            % calculate the range of index of height and width of inputs in
            % an extended (margin-inclusive) pad
            h_range = ceil(h_margin/2) : input_blob.get_height()+ceil(h_margin/2)-1;
            w_range = ceil(w_margin/2) : input_blob.get_width()+ceil(w_margin/2)-1;
            num_channels = input_blob.get_num_channels();
            % the extended input space
            pad = zeros(input_blob.get_height() + h_margin,...
                       input_blob.get_width() + w_margin,...
                       num_channels);
            pad(h_range, w_range, :) = input_blob.get_data();
           
            output_data = ones( ...
                ( size(pad, 1) - self.kernel_size) / self.stride , ...
                ( size(pad, 2) - self.kernel_size) / self.stride , ...
                num_channels);
            x = 1:self.stride:size(pad,1) - self.kernel_size + mod(input_blob.get_height(),self.stride);
            y = 1:self.stride:size(pad,2) - self.kernel_size + mod(input_blob.get_width(),self.stride);
            my_kernel_size = self.kernel_size;
            for i = 1:length(x)
                for j = 1:length(y)
                    % get a patch of kernel_size^2 in pad
                    patch_x = x(i):x(i)+my_kernel_size-1;
                    patch_y = y(j):y(j)+my_kernel_size-1;
                    onepatch = pad(patch_x, patch_y,:);
                    onepatch_columns = reshape(onepatch,[],num_channels); % reshape one channel into one column
                    [output_data(i, j, :), ~] = max(onepatch_columns, [], 1);
                end
            end
            % normalization
            parfor i = 1:num_channels
                nom = norm(output_data(:, :, i));
                if nom ~= 0
                    output_data(:, :, i) = output_data(:, :, i) / nom;
                end
            end
        end
        function [self, output_diff, delta_weights, delta_bias] = backward(self, input_blob)
            num_channels = input_blob.get_num_channels();
            my_kernel_size = self.kernel_size;
            % calculate the additional pixels needed for pooling on the
            % boundaries
            h_margin = self.stride - mod(input_blob.get_height(),self.stride);
            w_margin = self.stride - mod(input_blob.get_width(),self.stride);
            
            % calculate the range of index of height and width of inputs in
            % an extended (margin-inclusive) pad
            h_range = ceil(h_margin/2) : input_blob.get_height()+ceil(h_margin/2)-1;
            w_range = ceil(w_margin/2) : input_blob.get_width()+ceil(w_margin/2)-1;

            % create a pad and put the input data incide.
            pad = zeros(input_blob.get_height() + h_margin,...
                       input_blob.get_width() + w_margin,...
                       num_channels);
            pad(h_range, w_range, :) = input_blob.get_data();
            switches = zeros(size(pad));
            % index of the left top corner of a patch in pad
            x = 1:self.stride:size(pad,1) - self.kernel_size + mod(input_blob.get_height(),self.stride);
            y = 1:self.stride:size(pad,2) - self.kernel_size + mod(input_blob.get_width(),self.stride);
            
            for i = 1:length(x)
                for j = 1:length(y)
                    % get a patch of kernel_size^2 in pad
                    patch_x = x(i):x(i)+my_kernel_size-1;
                    patch_y = y(j):y(j)+my_kernel_size-1;
                    onepatch = pad(patch_x, patch_y,:);
                    onepatch_columns = reshape(onepatch, [], num_channels); % reshape one channel into one column
                    [~, index] = max(onepatch_columns, [], 1);
                    switch_mask = zeros(my_kernel_size^2, num_channels); % the same size as onepatch_columns
                    for k = 1:length(index)
                        switch_mask(index(k), k) = 1; % set true on the activated position 
                    end 
                    switches(patch_x, patch_y, :) = reshape(switch_mask, my_kernel_size, self.kernel_size, num_channels);
                end
            end
            
            % the extended input space
            pad = switches;
            % index of the left top corner of a patch in pad
            x = 1:self.stride:size(pad, 1) - self.kernel_size + mod(input_blob.get_height(),self.stride);
            y = 1:self.stride:size(pad, 2) - self.kernel_size + mod(input_blob.get_width(),self.stride);
            
            for i = 1:length(x)
                for j = 1:length(y)
                    % index range corresponding to x and y
                    patch_x = x(i):x(i)+self.kernel_size-1;
                    patch_y = y(j):y(j)+self.kernel_size-1;
                    % assign the diff to the activated cell, looping for
                    % every chanel
                    input_diff = input_blob.get_diff(i, j);
                    parfor k = 1:num_channels
                        pad(patch_x, patch_y, k) = pad(patch_x, patch_y, k) .* input_diff(:, :, k);
                    end
                end
            end 
            output_diff = pad(h_range, w_range, :);
            delta_weights = [];
            delta_bias = [];
        end
    end

end


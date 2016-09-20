classdef affine_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        num                % number of units
        num_input          % number of inputs
        units              % size -> num * num_inputs
        bias               % size -> num
        alpha              % learning rate
        %forward_input_data % for parameter update
%         units_delta        % updated weights to apply when updating
%         bias_delta         % updated bias to apply when updating
        dropout            % dropout mask, 0s are drop-out
        gpu                % weather to use gpu to compute
    end 
    
    methods (Access = public)
        function self = affine_layer(num_input, num, alpha)
            self@Layer('affine');
            self.num_input = num_input;
            self.num = num;
            rand_unit_real = normrnd(0, 1, num_input, num)*sqrt(2.0/num_input);
            rand_unit_imag = normrnd(0, 1, num_input, num)*sqrt(2.0/num_input);
            self.units = complex(rand_unit_real, rand_unit_imag);
            rand_bias_real = zeros(num, 1);
            rand_bias_imag = zeros(num, 1);
            self.bias = complex(rand_bias_real, rand_bias_imag);
            self.alpha = alpha;
            units_delta = complex(zeros(size(self.units)));
            self.dropout = ones(size(self.units));
            bias_delta = complex(zeros(size(self.bias)));
            self.gpu = 0;
        end
        function units = get_weights(self)
            units = self.units;
        end
        function bias = get_bias(self)
            bias = self.bias;
        end
        function [self, output_data] = forward(self, input_blob)
            assert(length(input_blob.get_data()) == self.num_input)
            % store inputs for backpropagation 
            input_data = reshape(input_blob.get_data(), self.num_input, 1);
            %self.forward_input_data = input_data;
            if self.gpu == 0
                output_data = (self.dropout .* self.units)' * input_data + self.bias;
                % normalize outputs
                output_data = output_data / norm(output_data);
            else
                % create data in gpu
                g_dropout = gpuArray(self.dropout);
                g_units = gpuArray(self.units);
                % calculate outputs
                g_units_dropin = arrayfun(@(a,b) (a.*b), g_dropout, g_units);
                units_dropin = gather(g_units_dropin);
                output_data = units_dropin' * input_data + self.bias;
                % normalize outputs
                output_data = gather(output_data / norm(output_data));
            end
   
        end
        
        function [self, output_diff, units_delta, bias_delta] = backward(self, input_blob)
            %hermitian conjugate 
            input_data = reshape(input_blob.get_data(), [], 1);
            input_diff = reshape(input_blob.get_diff(), self.num, 1);
            output_diff = reshape(conj(permute(self.units, [2 1]))' * input_diff, 1, 1, self.num_input);
            
            % calculate gradients of weights and bias respetively
            gradients_per_weights = input_diff * conj(permute(input_data, [2 1]));
            gradients_per_bias = input_diff;
            
            % replicate input data by num
            %forward_input_data_array = repmat(input_data, 1, self.num); 
            
            if self.gpu == 0
                %units_delta = self.alpha * ( forward_input_data_array .* gradients_per_weights' );
                units_delta = self.alpha * gradients_per_weights';
            else
                % create data in gpu
                g_gradients_per_weights = gpuArray(gradients_per_weights);
                g_forward_input_data_array = gpuArray(forward_input_data_array);
                g_alpha = self.alpha;
                % calculate delta
                g_units_delta = arrayfun(@(a,z,w) a*(z.*w) , ...
                    g_alpha, g_forward_input_data_array, g_gradients_per_weights.');
                units_delta = gather(g_units_delta);
            end
            bias_delta = self.alpha * gradients_per_bias;
        end
        
%         function self = set_delta(self, delta_units, delta_bias)
%             units_delta = delta_units;
%             bias_delta = delta_bias;
%         end
%         
%         function [delta_units, delta_bias] = get_delta(self)
%             delta_units = units_delta;
%             delta_bias = bias_delta;
%         end
        
        function self = update(self, units_delta, bias_delta)
            self.units = self.units + self.dropout .* units_delta;
            self.bias = self.bias + bias_delta;
%             units_delta = complex(zeros(size(self.units)));
%             bias_delta = complex(zeros(size(self.bias))); 
        end
        function self = set_dropout(self)
            self.dropout = rand(self.num_input, self.num) ./ sqrt(2/self.num_input) > 1;
        end
    end
end


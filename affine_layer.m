classdef affine_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        ReLU    % activation type
        num                % number of units
        num_input          % number of inputs
        units              % size -> num * num_inputs
        bias               % size -> num
        alpha              % learning rate
        forward_input_data % for parameter update
        forward_sum        % for gradient calculation
        units_delta        % updated weights to apply when updating
        bias_delta         % updated bias to apply when updating
    end 
    
    methods (Access = public)
        function self = affine_layer(ReLU, num_input, num, alpha)
            self@Layer('affine');
            self.ReLU = ReLU;
            self.num_input = num_input;
            self.num = num;
            rand_unit_real = ones(num_input, num);
            rand_unit_imag = ones(num_input, num);
            self.units = complex(rand_unit_real, rand_unit_imag);
            rand_bias_real = zeros(num, 1);
            rand_bias_imag = zeros(num, 1);
            self.bias = complex(rand_bias_real, rand_bias_imag);
            self.alpha = alpha;
            self.units_delta = complex(zeros(size(self.units)));
            self.bias_delta = complex(zeros(size(self.bias)));
        end
        function units = get_weights(self)
            units = self.units;
        end
        function [self, output_data] = forward(self, input_blob)
            assert(length(input_blob.get_data()) == self.num_input)
            input_data = reshape(input_blob.get_data(), self.num_input, 1);
            self.forward_input_data = input_data;
            output_data = self.units' * input_data + self.bias;
            self.forward_sum = output_data;
            %output_data = arrayfun(@self.activate, output_data);
            output_data = output_data / norm(output_data);
            %output_data = input_blob.set_data(output_data);
        end
        function [self, output_diff] = backward(self, input_blob)
%            switch self.ReLU
%                case 'ReLU'
%                     input_diff = sum(input_blob.get_diff());
%                     replicate into num of units
%                     input_diff = repmat(input_diff, self.num, 1);
%                 case 'SVM'
%                     input_diff = input_blob.get_diff();
%                     do nothing
%             end
%             
            % calculate the gradients in each unit
%            gradients_per_unit = arrayfun(@self.d_activate, input_diff, self.forward_sum);
            %hermitian conjugate 
            input_diff = reshape(input_blob.get_diff(), self.num, 1);
            output_diff = reshape(conj(permute(self.units, [2 1]))' * input_diff, 1, 1, self.num_input);
            
            % calculate gradients of weights and bias respetively
            gradients_per_weights = input_diff * conj(permute(self.forward_input_data, [2 1]));
            gradients_per_bias = input_diff .* self.bias;
            
            % replicate input data by num
            forward_input_data_array = repmat(self.forward_input_data, 1, self.num); 
            units_new = self.units - self.alpha * ( forward_input_data_array .* gradients_per_weights' );
            % limit the value of weights in
            % [sqrt(num), sqrt(num)]
            % compare in real value domain
            limit_units = ones(size(units_new))*sqrt(self.num);  
            units_real = real(units_new);
            units_imag = imag(units_new);
            units_real = max(min(units_real, limit_units), -limit_units);
            units_imag = max(min(units_imag, limit_units), -limit_units);
            self.units_delta = self.units_delta + complex(units_real, units_imag);
            
             
            bias_new = self.bias - self.alpha * gradients_per_bias;
            % limit the value of bias in
            % [sqrt(num), sqrt(num)]
            % compare in real value domain
            limit_bias = ones(size(bias_new))*sqrt(self.num);  
            bias_real = real(bias_new);
            bias_imag = imag(bias_new);
            bias_real = max(min(bias_real, limit_bias), -limit_bias);
            bias_imag = max(min(bias_imag, limit_bias), -limit_bias);
            self.bias_delta = self.bias_delta + complex(bias_real, bias_imag);
            
            % replicate into num_outputs by num_inputs
            %forward_input_data_array = repmat(self.forward_input_data, 1, self.num);
            %gradients_array = repmat(gradients_per_unit, 1, self.num_input)';
            % update weights
            %self.units = self.units - self.alpha * ( forward_input_data_array .* gradients_array );
            % update bias
            %self.bias = self.bias - self.alpha * gradients_per_unit;
            %output_diff = reshape(gradients_per_unit, 1, 1, self.num);
            %output_blob = input_blob.set_diff(output_data);
        end
        
        function self = update(self)
            self.units = self.units + self.units_delta;
            self.bias = self.bias + self.bias_delta;
            self.units_delta = complex(zeros(size(self.units)));
            self.bias_delta = complex(zeros(size(self.bias))); 
        end
    end
    % below is obsoleted owing to new findings
    methods (Access = private)
        function output = activate(self, input)
            switch self.ReLU
                case 'ReLU'
                   if real(input)>0 && imag(input)>0 
                       output = input;
                   else
                       output = complex(0,0);
                   end
                case 'SVM'
                   output = input;
            end        
        end
        function output = d_activate(self, input, forwarded_input)
            switch self.ReLU
                case 'ReLU'
                   % calculate the derivative of ReLU with inputs in
                   % forward-propagation
                   % decompose f = u + v
                   % refer to master thesis page 4.3.2
                   if real(forwarded_input) > 0 && imag(forwarded_input)> 0
                      dU = 1;
                      dV = 1;
                   else
                      dU = 0;
                      dV = 0;
                   end
                   output = real(input)*complex(dU, dV) + 1i*imag(input)*complex(dU, -dV);
                case 'SVM'
                   output = input;
            end
        end
    end
end


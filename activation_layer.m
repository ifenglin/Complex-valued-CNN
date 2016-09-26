classdef activation_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
    end 
    
    methods (Access = public)
        function self = activation_layer()
            % setup self handle and attributes
            self@Layer('activation');
        end
        
        function [self, output_data] = forward(self, input_blob)
            % ReLU
%             output_data = (real(input_blob.get_data())>0 & imag(input_blob.get_data())>0) .* input_blob.get_data();
            % tanh
            input_data = input_blob.get_data();
            output_data = tanh(real(input_data)) + 1i * tanh(imag(input_data));
        end
        function [self, output_diff, units_delta, bias_delta] = backward(self, input_blob)     
            units_delta = [];
            bias_delta = [];
            input_diff = input_blob.get_diff();
            % ReLU
%             derivative = double(real(input_blob.get_data())>0 & imag(input_blob.get_data())>0);
%             output_diff =  real(input_blob.get_data()) .* complex(derivative, derivative) + ...
%                 1i * imag(input_blob.get_data()) .* complex(derivative, -derivative);  
            %tanh
            output_diff = 1/2 * ( sech(real(input_diff))^2 - 1i * sech(imag(input_diff)^2 ));
        end
    end
    
    % below is obsoleted for performance issues
    methods (Access = private)
        function output = activate(self, input)
            switch self.ReLU
                case 'ReLU'
                   if real(input)>0 && imag(input)>0 
                       output = input;
                   else
                       output = complex(0,0);
                   end
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
                   % another formula 
                   %output = input*complex(R,-I);
            end
        end
    end  
end


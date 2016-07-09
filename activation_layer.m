classdef activation_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        ReLU    % activation type
        forward_input_data; % keep a copy of input data during forward propagation
    end 
    
    methods (Access = public)
        function self = activation_layer(ReLU)
            % setup self handle and attributes
            self@Layer('activation');
            if nargin == 1
                self.ReLU = ReLU;
            else
                self.ReLU = 'ReLU';
            end
        end
        function [self, output_data] = forward(self, input_blob)
            % keep a copy of input data for backward-propagation
            self.forward_input_data = input_blob.get_data();
            output_data = arrayfun(@self.activate, input_blob.get_data());
            %output_data = input_blob.set_data(output_data);
        end
        function [self, output_diff] = backward(self, input_blob)
            output_diff = arrayfun(@self.d_activate, input_blob.get_diff(), self.forward_input_data);
            %output_blob = input_blob.set_diff(output_data);
        end
    end
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
                   if real(forwarded_input) > 0 && imag(forwarded_input)> 0  
                      R = 1;
                      I = 1;
                   else
                      R = 0;
                      I = 0;
                   end
                   output = input*complex(R,-I);
            end
                
        end
    end 
    
end


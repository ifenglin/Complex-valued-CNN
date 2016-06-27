classdef activation_layer < Layer
    %UNTITLED6 Summary of self class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        ReLU    % activation type
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
        function output_blob = forward(self, input_data)
            output_data = arrayfun(@self.activate, input_data.get_data());
            output_blob = Blob(output_data);
        end
        function backward()
            % do something
        end
    end
    methods (Access = private)
        function output = activate(self, input)
            switch self.ReLU
                case 'ReLU'
                   if real(input)>0 && imag(input)>0 
                       output = input;
                   else
                       output = 0;
                   end
            end
                
        end
    end 
end


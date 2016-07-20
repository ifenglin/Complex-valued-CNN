classdef Layer < matlab.mixin.Heterogeneous
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        type
        params % blobs
    end
    
    methods
        function self = Layer(type)
            % setup self handle and attributes
            self.type = type;
            %for n = 1:length(blobs)
            %    self.params(n) = blobs(n);
            %end
        end
        function weights = get_weights(~)
            weights = [];
        end
        function layer_type = get_type(self)
            layer_type = self.type;
        end
        function forward(~)
            % do something
        end
        function backward(~)
            % do something
        end
    end
end


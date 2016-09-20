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
        function bias = get_bias(~)
            bias = [];
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
%         function self = get_delta(self)
%             % do something
%         end
%         function self = set_delta(self)
%             % do something
%         end
        function self = set_dropout(self)
            % do something
        end
        function self = update(self, ~, ~)
            % do something
        end
    end
end


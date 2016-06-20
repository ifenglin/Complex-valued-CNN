classdef Blob
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        data
    end
    
    methods
        function self = Blob(data)
            % CHECK(is_valid_handle(hBlob_blob), 'invalid Blob handle');
            % setup self handle
            self.data = data;
        end
        function data = get_data(self, rows, cols)
            if nargin < 3
                data = self.data;
            else
                data = self.data(rows, cols);
            end
        end
        function set_data(self, data)
            self.data = data;
        end
        function height = get_height(self)
            height = size(self.data, 1);
        end
        function width = get_width(self)
            width = size(self.data, 2);
        end
        function channels = get_num_channels(self)
            channels = size(self.data, 3);
        end
        function num = get_num(self)
            num = size(self.data, 4);
        end
    end
end
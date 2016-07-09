classdef Blob
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)
        data
        diff
    end
    
    methods
        function self = Blob(data, diff)
            if nargin == 1 
            % CHECK(is_valid_handle(hBlob_blob), 'invalid Blob handle');
            % setup self handle
                self.data = data;
            elseif nargin == 2
                self.data = data;
                self.diff = diff;
            end
        end
        function data = get_data(self, rows, cols, ch)
            if nargin == 1
                data = self.data;
            elseif nargin == 3
                data = self.data(rows, cols, :);
            elseif nargin == 4
                data = self.data(rows, cols, ch);
            end
        end
        function self = set_data(self, data)
            self.data = data;
        end
        function self = set_diff(self, diff)
            self.diff = diff;
        end
        function diff = get_diff(self, rows, cols, ch)
            if nargin == 1
                diff = self.diff;
            elseif nargin == 3
                diff = self.diff(rows, cols, :);
            elseif nargin == 4
                diff = self.diff(rows, cols, ch);
            end
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
function [data] = input_loader(input_file)
% input_loader(input_file)
%
% Load .mat file and examine the dimensions of the input matrix,
% which should be a 4-D Height x Width x Channel x Num complex matrx.
%
%
% input
%   input_file       the path to the .mat file that contains a variable
%                    named 'data'
%
% output
%   data             the loaded matrix
%
% std output
%
%   Height: the height of the images 
%   Width: the width of the images
%   Channel: the number of channels (RGB/polarization)
%   Num: the number of images
%   Real: 1 if the matrix is in real domain, 0 if in complex domain
%
%
im = load(input_file);
data = im.data;
str = {};
str{end+1} = sprintf('Height: %d', size(data,1));
str{end+1} = sprintf('Width: %d', size(data,2));
str{end+1} = sprintf('Channel: %d', size(data,3));
str{end+1} = sprintf('Num: %d', size(data,4));
str{end+1} = sprintf('Real: %d', isreal(data));
disp(str);
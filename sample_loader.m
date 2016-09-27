function data = sample_loader(allData, label, number, patch_size)
% sample_loader(allData, label, number)
% 
% Extracts a 64 x 64 image from the complex, three-channel data contained
% in "allData" around a point specified in "label"; "number" is the number
% of the label point and can be an array.
% 
% Depending on the length of "number", the resulting "data" will either be
% complex 64 x 64 x 3 or, if multiple images are requested, a complex
% 64 x 64 x 3 x lenght(number).
% 
% input
%   allData          the complete SAR data as loaded from alldata.mat
%   label            the label data, as loaded from e.g. city.mat
%   number           the number(s) of points from the label data, around
%                    which the image segment is requested
%
% output
%   data             the desired image segment; multidimensional if needed
% 
% example with the first 100 city points within the image
%   test_data = sample_loader(all_data, city, (1:100));
%   imshow(test_data(:,:,:,1))
%
% example with 100 random samples from within the image labeled as "city"
%
%   nums = randperm(length(city));
%   test_data = sample_loader(all_data, city, nums(1:100));
%   imshow(test_data(:,:,:,1))
if nargin < 4
    patch_size = 16;
end
x = label(number(1),1);
y = label(number(1),2);
half_size = ceil(patch_size/2);
data = allData(y-half_size+2:y+half_size+1, x-half_size+2:x+half_size+1, :);

if length(number) > 2
    for n = 2:length(number)
        x = label(number(n),1);
        y = label(number(n),2);
        data = cat(4, data, allData(y-half_size+2:y+half_size+1, x-half_size+2:x+half_size+1, :));
    end
end
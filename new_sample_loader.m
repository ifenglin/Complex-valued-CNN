function data = new_sample_loader(allData, x, y, patch_size)
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
half_size = ceil(patch_size/2);
for i = 1:length(x)
    my_data = allData(:,x(i)-half_size+1:x(i)+half_size,:);
    parfor j = 1:length(y)
        data(:,:,:,j,i) = my_data(y(j)-half_size+1:y(j)+half_size,:,:);
    end
end
data = reshape(data, size(data,1), size(data,2), size(data,3), []);
%% Image display

% replace all_data with whatever you want to show

R = abs(all_data(:,:,1));
G = abs(all_data(:,:,2));
B = abs(all_data(:,:,3));

% limit, change it to make the image brigher/darker
t = 5;

R(R>t)=t;
B(B>t)=t;
G(G>t)=t;

maxValue = double(max(max(R)));
minValue = double(min(min(R)));
rImage = 2 * (double(R) - (maxValue + minValue)/2)/(maxValue - minValue);

maxValue = double(max(max(G)));
minValue = double(min(min(G)));
gImage = 2 * (double(G) - (maxValue + minValue)/2)/(maxValue - minValue);

maxValue = double(max(max(B)));
minValue = double(min(min(B)));
bImage = 2 * (double(B) - (maxValue + minValue)/2)/(maxValue - minValue);

dImage = cat(3, rImage, gImage, bImage);

dImage = imrotate(dImage,-90);

imshow(dImage, []);
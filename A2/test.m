[B,map] = imread('digit.bmp');
B = double(255 - B(:,:,1)) / 255;
B = imgaussfilt(B)

image(B,'CDataMapping','scaled');
B = reshape(B, 1, 28*28);
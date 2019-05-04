function [] = dispRGB(img)
    img = reshape(img, 32, 32, 3) / 2^8;
    img = permute(img, [2, 1, 3]);
    image(img);
end
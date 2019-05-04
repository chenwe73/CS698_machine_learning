function [] = displayImage(X)
    w = 28;
    A = reshape(X, w, w);
    image(A,'CDataMapping','scaled');
end

function [Y] = processImage(X)
    
    Y = X;
    [n, d] = size(X);
    w = 28;
    for i = 1:n
        A = reshape(X(i, :), w, w);
        A = imgaussfilt(A);
        Y(i, :) = reshape(A, 1, w*w);
    end
    
end
% Euclidean distance matrix from each X to Z
function [D] = dist(X, Z)

    XX = sum(X.^2, 2);
    ZZ = sum(Z.^2, 2);
    D = bsxfun(@plus, XX, ZZ') - 2 * X * Z';
    
end
% k nearest neighbour
% Z is test X
function [yHat] = kNN(X, y, Z, k)

    [n, d] = size(X);
    m = size(Z,1);
    D = zeros(n,m);
    yHat = zeros(m,1);
    
    % partition Z into blocks
    blockSize = 2000;
    from = 1;
    while (from <= m)
        to = from + blockSize - 1;
        if (to > m)
            to = m;
        end
        %[from to]
        
        % compute distance
        Zpart = Z(from:to, :);
        [D] = dist(X, Zpart);
        
        % find results and vote
        for t = 1:(to - from +1)
            [~, min_i] = mink(D(:, t), k);
            yHat(from + t - 1) = mode(y(min_i));
        end
        from = to + 1;
    end
    
end

% Euclidean distance matrix from each X to Z
function [D] = dist(X, Z)
    XX = sum(X.^2, 2);
    ZZ = sum(Z.^2, 2);
    D = bsxfun(@plus, XX, ZZ') - 2 * X * Z';
end

% minimum k values
function [B,I] = mink(A,k)
    temp = A;
    B = zeros(k,1);
    I = zeros(k,1);
    for i = 1:k
        [B(i), I(i)] = min(temp);
        temp(I(i)) = inf;
    end
end
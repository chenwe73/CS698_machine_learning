% k nearest neighbour
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
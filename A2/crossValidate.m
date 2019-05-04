function [kBest, errPercent] = crossValidate(X, y, k)
    
    [n, d] = size(X);
    kSize = size(k, 2);
    
    FOLD = 10;
    setSize = ceil(n / FOLD);
    errPercent = zeros(1, kSize);
    
    for i = 1:kSize
        for j = 1:FOLD
            from = (j-1) * setSize + 1;
            to = j * setSize;
            if (to > n)
                to = n;
            end
            %validRange = [from to]
            % get result from one valid set
            trainIndx = [1:from-1 to+1:n];
            validIndx = [from:to];
            validNum = to - from +1;
            [yHat, ~] = kNN(X(trainIndx, :), y(trainIndx, :), X(validIndx, :), k(i));
            % compute error
            err = (yHat ~= y(validIndx, :));
            errSet = sum(err) / validNum * 100;
            errPercent(i) = errPercent(i) + errSet / FOLD;
        end
        k_err = [k(i) errPercent(i)]
    end

    [minErr, iBest] = min(errPercent);
    kBest = k(iBest);
end


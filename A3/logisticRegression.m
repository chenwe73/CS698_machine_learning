% Newton's method
% w = w - hessian(f(w))^-1 . gradient(f(w))
% y = -1 or +1
function [w] = logisticRegression(X, y, lambda, isKernal)
    MAX_ITERATION = 30;
    MIN_dW = 1e-3; % 1e-3
    
    [Xpos, Xneg] = seperate(X, y);
    
    K = eye(size(X, 2));
    if (isKernal)
        K = X;
    end
    
    [nPos, d] = size(Xpos);
    [nNeg, d] = size(Xneg);
    w = zeros(1, d);
    i = 0;
    dw = inf;
    
    while (i < MAX_ITERATION && dw > MIN_dW)
        pPos = sigmoid(Xpos * w');
        pNeg = sigmoid(-Xneg * w');
        
        gradientPos = gradient(nPos, Xpos, pPos);
        gradientNeg = -gradient(nNeg, Xneg, pNeg);
        gradient = gradientPos + gradientNeg + 2 * lambda * K * w';
        
        hessianPos = hessian(nPos, Xpos, pPos);
        hessianNeg = hessian(nNeg, Xneg, pNeg);
        hessian = hessianPos + hessianNeg + 2 * lambda * K;
        
        % update
        v = hessian \ gradient;
        w = w - v';
        dw = norm(v);
        i = i + 1;
        %disp([i, dw]);
    end
end


function [g] = gradient(n, X, p)
    g = 1/n * X' * (p-1);
end

function [h] = hessian(n, X, p)
    h = 1/n * X' * (p .* (1-p) .* X);
end

function [pos, neg] = seperate(X, y)
    iPos = find(y == 1);
    iNeg = find(y == -1);
    pos = X(iPos, :);
    neg = X(iNeg, :);
end


function [w] = kernelLogisticReg(X, y, lambda)
    MAX_ITERATION = 30;
    MIN_dW = 1e-3; % 1e-3
    
    [n, d] = size(X);
    w = zeros(1, d);
    i = 0;
    dw = inf;
    
    while (i < MAX_ITERATION && dw > MIN_dW)
        p = sigmoid(X * w');
        gradient = 1/n * X' * (p-(y+1)/2) + 2 * lambda * X * w';
        hessian = 1/n * X' * (p .* (1-p) .* X) + 2 * lambda * X;
        
        % update
        v = hessian \ gradient;
        w = w - v';
        dw = norm(v);
        i = i + 1;
        disp([i, dw]);
    end
end
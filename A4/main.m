% Wei Tao Chen
clear;

[X, K, p_init, mu_init, S_init] = testCase_1();
% [X, K, p_init, mu_init, S_init] = testCase_2();
[p, mu, S, l] = GMM_EM(X, K, p_init, mu_init, S_init);
[guess] = kNN(train_X, train_y, test_X, 3);

% for i = 5
%     K = i;
%     [train_X, train_y, test_X, test_y, classN] = loadData();
%     [Xc, p_init, mu_init, S_init, q] = init(train_X, train_y, classN, K);
% 
%     for c = 1:classN
%         disp(c);
%         [p{c}, mu{c}, S{c}, l{c}] = GMM_EM(Xc{c}, K, p_init, mu_init, S_init);
%     end
% 
%     [guess] = evaluate(test_X, p, mu, S, K, q, classN);
%     [error(i)] = compareError(guess, test_y);
% end
% plot(error);

function [Xtrain, ytrain, Xtest, ytest, classN] = loadData()
    classN = 10;
    load('MNIST_X_train.mat');
    load('MNIST_y_train.mat');
    load('MNIST_X_test.mat');
    load('MNIST_y_test.mat');
    
%     N = 1000;
%     Xtrain = Xtrain(1:N, :);
%     ytrain = ytrain(1:N);
%     Xtest = Xtest(1:N, :);
%     ytest = ytest(1:N);
end

function [Xc, p_init, mu_init, S_init, q] = init(train_X, train_y, classN, K)
    [Xc, yc, q] = split(train_X, train_y, classN);
    
    [~, d] = size(train_X);
    
    S_init = ones(K, d) * 1;
    p_init = ones(1, K) / K;
    
    % how to better initialize mu?
    mu_init = rand(K, d);
    
%     mu_init = double(rand(K, d) < 0.5);

%     mu_init = zeros(K, d);
%     div = floor(d/K);
%     for k = 1:K
%         mu_init(k, :) = [zeros(1, (k-1)*div), ones(1,div), zeros(1, (K-k)*div), 0, 0, 0, 0];
%     end
end

function [Xc, yc, q] = split(X, y, classN)
    for c = 1:classN
        digit = c - 1;
        i_digit = find(y == digit);
        Xc{c}(:, :) = X(i_digit, :);
        yc{c} = y(i_digit);
        q(c) = sum(y == digit);
    end
    q = q ./ sum(q);
end

function [X, K, p_init, mu_init, S_init] = testCase_1()
%     X = [10 9 8 90 91]';
    K = 2;
    
    mu_init = [0; 1];
    S_init = [0.9; 0.3];
    p_init = [0.6 0.4];
    
    mu = [0.3; 0.8];
    S = [0.001; 0.01];
    p = [0.3 0.7];
    N = 100
    X = [];
    for k = 1:K
        Xk = normrnd(mu(k), sqrt(S(k)), p(k)*N, 1);
        X = [X; Xk];
    end
    
%     p_init = p;
%     mu_init = mu;
%     S_init = S;
end

function [X, K, p_init, mu_init, S_init] = testCase_2()
    X = [2 2; 3 3; 2 3; 3 2; 8 8; 7 8; 8 7; 7 7] / 10;
    K = 2;
    
    mu_init = [0 0; 1 1];
    S_init = [1 1; 1 1];
    p_init = [0.5 0.5];
end

function [guess] = evaluate(X, p, mu, S, K, q, classN)
    for c = 1:classN
        for k = 1:K
            log_r(:, k) = logGaussian(X, p{c}(k), mu{c}(k,:), S{c}(k,:));
        end
        log_rc(:,c) = q(c) * sum(log_r, 2);
    end
    
    [M, I] = max(log_rc, [], 2);
    guess = I - 1;
end

function [error] = compareError(guess, answer)
    error = sum(guess ~= answer) / size(answer, 1) * 100;
end


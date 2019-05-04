% Wei Tao Chen
clear;

lambda = 1;

[X, y, test_X, test_y, classN] = init2();

[guessLabel, p] = oneVsAll(X, y, test_X, lambda, classN);
error = difference(guessLabel, test_y)

[guessLabel, p] = oneVsOne(X, y, test_X, lambda, classN);
error = difference(guessLabel, test_y)

[X, y, test_X, test_y, classN] = init();

K = kernel_linear(X, X);
Ktest = kernel_linear(X, test_X);
p = 5
K = kernel_poly(X, X, p);
Ktest = kernel_poly(X, test_X, p);
sigma = 100;
K = kernel_gaussian(X, X, sigma);
Ktest = kernel_gaussian(X, test_X, sigma);

alpha = logisticRegression(K, y, lambda, 1);
[errorTrain, ~] = evaluateLogistic(K' * (alpha'), y);
[errorTest, guess] = evaluateLogistic(Ktest' * alpha', test_y);



% w = logisticRegression(X, y, lambda, 0);
% [errorTrain, ~] = evaluateLogistic(X * w', y);
% [errorTest, guess] = evaluateLogistic(test_X * w', test_y);

% w2 = (X\y)';
% [error, guess] = evaluateLinear(test_X, test_y, w2);

% guess = kNN(X, y, test_X, 3);
% e = difference(guess, test_y)

% dispRGB(X(1,1:3073));

function [X, y, test_X, test_y, classN] = init2()
    classN = 10;
    X = [];
    y = [];
    
    load('cifar-10-batches-mat/data_batch_1.mat');
    X = [X; data(:, :)];
    y = [y; labels(:)];
%     load('cifar-10-batches-mat/data_batch_2.mat');
%     X = [X; data];
%     y = [y; labels];
%     load('cifar-10-batches-mat/data_batch_3.mat');
%     X = [X; data];
%     y = [y; labels];
%     load('cifar-10-batches-mat/data_batch_4.mat');
%     X = [X; data];
%     y = [y; labels];
%     load('cifar-10-batches-mat/data_batch_5.mat');
%     X = [X; data];
%     y = [y; labels];
    X = double(appendOnes(X));
    y = double(y);
    
    load('cifar-10-batches-mat/test_batch.mat');
    test_X = double(appendOnes(data));
    test_y = double(labels);
end

function [guessLabel, vote] = oneVsOne(X, y, test_X, lambda, classN)
    vote = zeros(size(test_X, 1), classN);
    for i = 1:classN
        for j = i+1:classN
            % get w of class i and j
            [Xij, yij] = OVOclass (X, y, i, j);
            w = logisticRegression(Xij, yij, lambda, 0);
            % training error (optional)
            [errorTrain, ~] = evaluateLogistic(Xij * w', yij);
            disp([i errorTrain]);
            % vote
            p = sigmoid(test_X * w');
            vote(:, i) = vote(:, i) + (p >= 0.5);
            vote(:, j) = vote(:, j) + (p < 0.5);
        end
    end

    [~, guessI] = max(vote');
    guessLabel = guessI';
end

% i is +1, j is -1, others are 0
function [Xij, yij] = OVOclass (X, y, i, j)
    yi = OVAclass(y, i, 0);
    yj = OVAclass(y, j, 0);
    yij = yi - yj;
    iNot0 = find(yij ~= 0);
    yij = yij(iNot0);
    Xij = X(iNot0, :);
end

function [guessLabel, p] = oneVsAll(X, y, test_X, lambda, classN)
    for i = 1:classN
        label = i;
        yi = OVAclass(y, label, -1);
        w = logisticRegression(X, yi, lambda, 0);
        % training error (optional)
        [errorTrain, ~] = evaluateLogistic(X * w', yi);
        disp([i errorTrain]);

        p(:, i) = sigmoid(test_X * w');
    end

    [~, guessI] = max(p');
    guessLabel = guessI';
end

function [y] = OVAclass(x, labelAsOne, yRest)
    y = (x == labelAsOne) + yRest * (x ~= labelAsOne);
end

function [X, y, test_X, test_y, classN] = init()
    classN = 2;
    load('train_dog_cat.mat');
    load('test_dog_cat.mat');
    X = appendOnes(train_X / 2^8);
    y = train_y;
    
    test_X = appendOnes(test_X / 2^8);

    [X, y] = shuffleRow(X, y);
    % N = 100;
    % X = X(1:N, :);
    % y = y(1:N);
end

function [Y] = appendOnes(X)
    Y = [X, ones(size(X, 1), 1)];
end

function [A, B] = shuffleRow(X, Y)
    [n, d] = size(X);
    idx = randperm(n);
	A = X;
    B = Y;
    A(:, :) = X(idx, :);
    B(:, :) = Y(idx, :);
end

function [error, guess] = evaluateLogistic(Z, y)
    p = sigmoid(Z);
    guess = (p >= 0.5) - (p < 0.5);
    error = difference(guess, y);
end

function [error, guess] = evaluateLinear(X, y, w)
    p = X * w';
    %guess = (p >= 0) - (p < 0);
    guess = round(p);
    error = difference(guess, y);
end

function [e] = difference(x, y)
    n = size(y, 1);
    e = sum(x ~= y) / n * 100;
end

function [K] = kernel_linear(X1, X2)
    K = (X1 * X2');
end

function [K] = kernel_poly(X1, X2, p)
    K = (X1 * X2' + 1).^p;
end

function [K] = kernel_gaussian(X1, X2, sigma)
    n = size(X1, 1);
    m = size(X2, 1);
    X11 = repmat(sum(X1.^2,2), 1, m);
    X22 = repmat(sum(X2.^2,2)', n, 1);
    Z = X11 - 2 * X1 * X2' + X22;
    K = exp(-Z / sigma);
end


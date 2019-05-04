%% Wei Tao Chen
clear;

filename = 'MNIST_X_train.mat';
m = matfile(filename);
Xtrain = m.Xtrain;

filename = 'MNIST_y_train.mat';
m = matfile(filename);
ytrain = m.ytrain;

filename = 'MNIST_X_test.mat';
m = matfile(filename);
Xtest = m.Xtest;

filename = 'MNIST_y_test.mat';
m = matfile(filename);
ytest = m.ytest;

% Xtrain = processImage(Xtrain);
% Xtest = processImage(Xtest);
[B,~] = imread('digit.bmp');
[myImageGuess] = testImage(B, Xtrain, ytrain)

n = 6000;
m = 1000;

%% cross-validation
k = [1 3 5 9 17 25];
[kBest, errPercentValid] = crossValidate(Xtrain(1:n,:), ytrain(1:n,:), k);
plot(k, errPercentValid);

%% use test set
kBest = 7;
[yHat, ~] = kNN(Xtrain(1:n,:), ytrain(1:n,:), Xtest(1:m,:), kBest);
err = (yHat ~= ytest(1:m,:));
errPercentTest = sum(err) / m * 100

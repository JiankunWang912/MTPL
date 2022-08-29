%% This is the MATLAB code for the following paper:
%
%   Multi-Task Personalized Learning with Sparse Network Lasso
%
%   Please run 'MTPL_sarcos.m' to conduct experiment on the real-world data set SARCOS.
%
%%
clc;
clear;
rng('default');
addpath('utils');
addpath('eval');

%% Load data
dataset = "./data/SARCOS/sarcos_01.mat";
[X_train, Y_train, X_validation, Y_validation, X_test, Y_test] = loadData(dataset);
numT = length(X_train);

%% Data Processing
final_Xtr = cellfun(@cat,mat2cell(2*ones(numT,1),ones(numT,1)),X_train,X_validation,'UniformOutput',false);
final_Ytr = cellfun(@cat,mat2cell(2*ones(numT,1),ones(numT,1)),Y_train,Y_validation,'UniformOutput',false);
final_Xte = X_test;
final_Yte = Y_test;
for t = 1:numT % standardize
    [final_Xtr{t},PS] = mapminmax(final_Xtr{t},0,1);
    % PS: Process settings that allow consistent processing of values
    final_Xte{t} = mapminmax('apply',final_Xte{t},PS);
end

%% Build the similarity graph
numNeighbor = 5;
final_Str = buildSimilarityGraph(final_Xtr, numNeighbor);

%% Set optimization parameters
opts.lambda1 = 2^-10;   % regularizaiton paramter \lambda_1
opts.lambda2 = 2^6;     % regularizaiton paramter \lambda_2
opts.lambda3 = 2^-6;    % regularizaiton paramter \lambda_3
opts.init     = 1;      % 1: guess start point from data 0: random;
opts.nIterIn  = 1000;   % number of iterations Inner loop
opts.nIterOut = 500;    % number of iterations Outside loop
opts.absTol   = 10^-3;  % termination condition
opts.dbFlag   = true;   % debug information (true: display; false: nothing)
opts.flagEta  = 'line'; % line search or fixed  for stepsize in proximal gradient descent
opts.eta      = 10^-5;  % value of fixed stepsize eta
opts.numK     = 11;     % number of latent topics

%% Build model using the optimization parameters
[learned_theta, STATS, learned_A, learned_B, learned_G] = Least_MTPL(final_Xtr, final_Ytr, final_Str,...
    opts.lambda1, opts.lambda2, opts.lambda3, opts.numK, opts);
[rmse,Rsquare,nmse,mae] = evalMTPL( final_Xte, final_Yte, final_Xtr, learned_theta);

%% Output
disp(['sarcos', ' rMSE: ',num2str(rmse),' nmse: ',num2str(nmse),' Rsquare: ',num2str(Rsquare),' mae: ',num2str(mae)]);
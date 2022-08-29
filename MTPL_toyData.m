%% This is the MATLAB code for the following paper:
%
%   Multi-Task Personalized Learning with Sparse Network Lasso
%
%   Please run 'MTPL_toyData.m' to demostrate the training of MTPL on the synthetic data set .
%
%%
%clc;
clear;
rng('default');
%% load data 
load_data = load('./data/toyData/toyData.mat'); 
data = load_data.data;
target = load_data.target;
S = load_data.S;
designedTheta = load_data.theta;
designedGt = load_data.Gt;
%% data parameters
numGs = 25; % number of group size
numT = size(data,1);
numD = size(data{1},1);
numN = zeros(numT,1);
for i = 1:numT
    numN(i) = size(data{i},2);
end
%% Set optimization parameters
opts.lambda1  = 2^2;     % regularizaiton paramter \lambda_1
opts.lambda2  = 2^8;     % regularizaiton paramter \lambda_2
opts.lambda3  = 2^6;     % regularizaiton paramter \lambda_3
opts.init     = 0;       % 0: random starting point; 1: warm up;;
opts.nIterIn  = 1000;    % number of iterations Inner loop
opts.nIterOut = 500;     % number of iterations Outside loop
opts.absTol   = 10^-5;   % termination condition
opts.dbFlag   = true;    % debug information (true: display; false: nothing)
opts.flagEta  = 'line';  % line search or fixed  for stepsize in proximal gradient descent
opts.eta      = 10^-5;   % value of fixed stepsize eta
opts.numK     = 6;       % number of latent topics
%% Build model using the optimal parameter 
[learned_theta, STATS, learned_A, learned_B, learned_G] = Least_MTPL(data, target, S,...
    opts.lambda1, opts.lambda2, opts.lambda3, opts.numK, opts);
%% Illustrate results
figure;
f1=tiledlayout(numT/2,2,'TileSpacing','compact');
for i=1:numT
    nexttile;
    imagesc(designedGt{i}(:,2:end));
    title(['Task ', num2str(i)],'FontName','Times New Roman','FontSize',20 );
    colormap(flipud(gray));
    a = get(gca,'xticklabel');
    set(gca,'xticklabel',a,'FontName','Times New Roman','fontsize',20);
end
cb1 = colorbar;
cb1.Layout.Tile = 'west';
cb1.FontName='Times New Roman';
cb1.FontSize=20;
figure;
f2=tiledlayout(numT/2,2,'TileSpacing','compact');
for i=1:numT
    nexttile;
    imagesc(learned_G{i}(:,2:end));
    title(['Task ', num2str(i)],'FontName','Times New Roman','FontSize',20 );
    colormap(flipud(gray));
    a = get(gca,'xticklabel');
    set(gca,'xticklabel',a,'FontName','Times New Roman','fontsize',20);
end
cb2 = colorbar;
cb2.Layout.Tile = 'east';
cb2.FontName='Times New Roman';
cb2.FontSize=20;
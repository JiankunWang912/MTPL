function [rmse,Rsquare,nmse,mae]  = evalMTPL(Xte, Yte, Xtr, theta)
%% FUNCTION eval_MTPL
%   Compute rmse, Rsquare, nmse, mae.
%   The values of rmse, nmse and mae are the lower the better,
%   while the value of Rsquare is the larger the better.
%
%% OUTPUT
%
%  rmse = ( sum_t(rmse(t) * N_t) ) / sum_t N_t
%  mae = ( sum_t(mae(t) * N_t) ) / sum_t N_t
%  Rsquare = ( sum_t(rsquare(t)) ) / sum_t N_t
%  nmse = ( sum_t(nmse(t)) ) / sum_t N_t
%
%  where 
%     rmse(t) = sqrt(sum((Y_pred - Y_true)^2)/ N_t)
%     mae(t) = sum(abs(Y_pred - Y_true))/N_t
%     nmse(t) = (sum((Y_pred - Y_true)^2)/ N_t)/sqrt(sum((Y_true).^2))
%     rsquare(t) = 1 - sum((Y_true-Y_pred)^2)/sum((Y_true-mean(Y_true_all))^2)
%     Y_pred(n) = X{t}(:,n) * theta{t}(:,n)
%     N_t     = length(Y{t})
%
%% INPUT
%   Xte: {d * n} * t
%   Yte: {1 * n} * t
%   Xtr: {d * n} * t
%   theta: {d * n} * t - weight 
%
    numT = size(Yte,1);
    rmse_values = zeros(numT,1);
    nmse_values = zeros(numT,1);
    rsquare_values = zeros(numT,1);
    mae_values = zeros(numT,1);
    k = 5;
    calWeight = @(edgeDistance) exp(-(0.01).*edgeDistance);
    totalsample = 0;
    for t = 1: numT
        findTheta = @(index) theta{t}(:,index);
        [neighbors, distances] =  knnsearch(Xtr{t}', Xte{t}', 'K', k, 'IncludeTies',true);
        neighborTheta = cellfun(findTheta,neighbors,'UniformOutput',false);
        neighborWeight = cellfun(calWeight,distances,'UniformOutput',false);
        thetaHat  = cellfun(@weberSolver,neighborTheta, neighborWeight,'UniformOutput',false);
        squareError = 0;
        absError = 0;
        numN = size(Yte{t},2);
        totalsample = totalsample + numN;
        for n=1:numN
            y_pred = (Xte{t}(:,n))' * thetaHat{n};
            squareError = squareError + (y_pred-Yte{t}(n))^2;
            absError = absError + abs(y_pred-Yte{t}(n));
        end
        mse = squareError/numN;
        mae = absError/numN;
        rmse_values(t) = ( sqrt(mse) ) * numN;
        mae_values(t) = mae * numN;
        nmse_values(t) = ( mse / var(Yte{t}) )* numN;
        rsquare_values(t) = ( 1 - ( mse / var(Yte{t}) ) ) * numN;
    end
    rmse = sum(rmse_values)/totalsample;
    mae = sum(mae_values)/totalsample;
    Rsquare = sum(rsquare_values)/totalsample;
    nmse = sum(nmse_values)/totalsample;
end

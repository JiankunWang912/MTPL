%% FUNCTION weberSolver
% 
%  To make the prediction on an unseen testing sample, we can solve
%  the Weber problem to estimate its peronalized model. 
%  Details are provided in the supplement.
%
%  Url: https://github.com/JiankunWang912/MTPL/blob/main/sup/
%
function thetaHat = weberSolver(neighborTheta, neighborWeight)
    [ numD, numN ] = size( neighborTheta );
    absTol = 10^-5;
    thetaHat = zeros(numD,1);
    FvalWeber = [];
    for iter = 1:200
        dist2 = sqrt( sum( ( repmat(thetaHat,1,numN) - neighborTheta ).^2, 1 ) );
        invdist2 = neighborWeight./(dist2+eps);
        suminvdist2 = sum(invdist2);
        thetaHat =( neighborTheta * invdist2' ) / suminvdist2;
        FvalWeber = cat(1, FvalWeber, neighborWeight * dist2');
        if iter>1 && abs(FvalWeber(iter)-FvalWeber(iter-1))<absTol * FvalWeber(iter-1)
            break;
        end
    end
end
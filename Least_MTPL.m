%% FUNCTION Least_MTPL
%   Multi-Task Personalized Learning with Least Squares Loss.
%
%% OBJECTIVE
%   argmin_{A,Bt,Gt} sum_{t,i}(y_ti - x_ti^T*theta_ti)^2 +
%   lambda1*(||A||_F^2 + sum_t||Bt||_F^2) +
%   lambda2*sum_t sum_{i,j}St_{ij}||g_ti-g_tj||_2 +
%   lambda3*sum_t||Gt||_1
%   s.t. theta_ti=(A+Bt)(g_t0+g_ti)
%
%% INPUT
%           X       T*1 labeled data cell
%           Y       T*1 target cell
%           S       T*1 similarity matrix for the data
%           lambda1: regularized parameter
%           lambda2: regularized parameter
%           lambda3: regularized parameter
%           numK : number of latent topics
%           opts : optimization parameters of MTPL
%
%% OUTPUT
%           theta: model weight (T*1 cell)
%           STATS: function values
%           A: task common weight (d*K matrix) 
%           B: task specific weight (T*1 cell) each cell is a d*K matrix 
%           G: Local model coefficients (T*1 cell ) each cell is a K*(Nt+1) matrix 
%             g_{t,0} is the t-th task's samples shared coefficient
%             g_{t,i} is the t-th task's specific coefficient for sample i
%



function [theta, STATS, A, B, G] = Least_MTPL(X, Y, S, lambda1, lambda2, lambda3, numK, opts)

if nargin < 8
    error('\n Inputs: X, Y, S,and regularized parameters should be specified!\n');
end

%% Get optimization parameters
nIterIn  = opts.nIterIn;
nIterOut = opts.nIterOut;
absTol   = opts.absTol;
dbFlag   = opts.dbFlag;
flagEta  = opts.flagEta;

%% Data statistics
numT = length(X);
numD = size(X{1}, 1);
numN = zeros(numT, 1);
for t = 1:numT
    numN(t) = size(X{t}, 2);
end
Z = cell(numT,1);
for t = 1:numT
    temp = mat2cell(X{t},numD,ones(1,numN(t)));
    temp = cellfun(@transpose,temp,'UniformOutput',false);
    Z{t} = sparse(blkdiag(temp{:}));
    Y{t} = Y{t}';
end

%% stepsize 
switch flagEta
    case 'fixed'
        fixed_eta= opts.eta;     %fixed
        initial_eta = fixed_eta;
    case 'line'
        initial_eta = 1;         %line search
end

%% auxiliary matrix
M = cell(numT ,1);
N = cell(numT ,1);
L = cell(numT ,1);
for t = 1:numT
    M{t} = sparse(cat(1,ones(1,numN(t)),eye(numN(t))));
    N{t} = sparse(cat(1,zeros(1,numN(t)),eye(numN(t))));
end

%% initialize starting points
rng('default');
B0 = cell(numT,1);
G0 = cell(numT,1);
if opts.init==1
    W = warmUp(X, Y);
    [U,~,~] = svd(W);
    A0 = U(:,1:numK);
    for t=1:numT
        B0{t} = zeros(numD,numK);
        G0{t} = normrnd(0,1, numK, numN(t)+1);
    end
elseif opts.init == 0
    A0 = normrnd(0,1,numD,numK);
    for t=1:numT
        B0{t} = normrnd(0,1,numD,numK);
    end
    for t=1:numT
        G0{t} = normrnd(0,1, numK, numN(t)+1);
    end
end

%% Iterative algorithm
A=A0;B=B0;G=G0;
FvalABG = calFvalABG(Z,Y,A,B,G,S,M,lambda2,lambda3,lambda1);
for iter_out = 1 : nIterOut

    % Step 1: Fix A B, Update G (for every task)
    %
    % Here, we introduce auxiliary variable L={l_ij}, according to
    % Black-Rangarjan Duality, the problem changes to the following form :
    % argmin_{Gt,L} sum_{i}(y_{t,i]-x_{t,i}^T(A+Bt)(g_{t,0}+g_{t,i}))^2+
    % lambda2*sum_{i,j}St_{i,j}(L_{i,j}||g_{t,i}-g_{t,j}||_2^2+1/(4*L_{i,j}))+
    % lambda3*||Gt||_1
    %

    for t = 1:numT
        FvalGL = [];
        for iter_in = 1 : nIterOut
            % Fix G, Update L
            Gtemp = (G{t}(:,2:(numN(t)+1)))';
            L{t} = 1./(2.*pdist2(Gtemp,Gtemp)+eps);
            % Fix L, Update G
            t_new = 1;
            G_hat = G{t};
            eta = initial_eta;
            FvalG = [];
            for iter_inn = 1 : nIterIn
                G_old = G{t};
                t_old = t_new;
                [grad_G,LL] = calGradG(Z{t},Y{t},A,B{t},G_hat,lambda2,S{t},L{t},M{t},N{t});
                switch flagEta
                    case 'line'
                        while 1
                            G{t} = prox_l1(G_hat-eta*grad_G,eta*lambda3);
                            GtN = G{t}*N{t};
                            GhatN = G_hat*N{t};
                            lossGt = calLoss(Z{t},Y{t},A,B{t},G{t},M{t});
                            lossGhat = calLoss(Z{t},Y{t},A,B{t},G_hat,M{t});
                            if (lossGt + 2*lambda2*trace(GtN*LL*GtN')) <= (lossGhat + 2*lambda2*trace(GhatN*LL*GhatN'))...
                                    + sum(dot(grad_G,G{t}-G_hat,1)) + (sum(sum((G{t}-G_hat).^2))/(2*eta))
                                break;
                            else
                                eta = eta / 2;    % /2 /5 /10
                            end
                        end
                    case 'fixed'
                        eta = fixed_eta;
                        G{t} = prox_l1(G_hat-eta*grad_G,eta*lambda3);
                end
                FvalG = cat(1,FvalG,calFvalG(Z{t},Y{t},A,B{t},G{t},LL,M{t},N{t},lambda2,lambda3));
                if iter_inn>1 && abs(FvalG(iter_inn)-FvalG(iter_inn-1))<absTol * FvalG(iter_inn-1)
                    break;
                end
                t_new = (1+sqrt(1+4*t_old^2))/2;
                G_hat = G{t} + ((t_old-1)/t_new)*(G{t}-G_old);
            end
            FvalGL = cat(1,FvalGL,calFvalGL(Z{t},Y{t},A,B{t},G{t},S{t},L{t},M{t},lambda2,lambda3));
            if iter_in>1 && abs(FvalGL(iter_in)-FvalGL(iter_in-1))<absTol * FvalGL(iter_in-1)
                break;
            end
        end
    end
    
    % Step 2: Fix A G, Update B (for every task)
    Id = speye(numD);
    Idk = speye(numD*numK);
    sumtttemp = zeros(numD*numK);
    sumyttemp = zeros(numD*numK,1);
    sumbttemp = zeros(numD*numK,1);
    for t = 1:numT
        temp = kron((G{t}*M{t})',Id);
        ttemp = Z{t}*temp;
        tttemp = ttemp' * ttemp;
        yttemp = ttemp'*Y{t};
        sumtttemp = sumtttemp + tttemp; % for updating A
        sumyttemp = sumyttemp + yttemp;
        vecBt = (tttemp +  lambda1*Idk) \ ( yttemp- tttemp*A(:));
        %vecBt = lsqminnorm( tttemp +  lambda1*Idk ,  yttemp- tttemp*A(:) );
        sumbttemp = sumbttemp + tttemp*vecBt;
        B{t} = reshape(vecBt,[numD,numK]);
    end
    
    % Step 3: Fix B G, Update A
    vecA = (sumtttemp + lambda1*Idk) \ (sumyttemp - sumbttemp);
    %vecA = lsqminnorm( sumtttemp + lambda1*Idk , sumyttemp - sumbttemp );
    A = reshape(vecA,[numD,numK]);
    
    % Calculate the objective function value 
    FvalABG = cat(1,FvalABG,calFvalABG(Z,Y,A,B,G,S,M,lambda2,lambda3,lambda1));
    % Output debug information if necessary
    if dbFlag
        disp([num2str(iter_out),'th iter, ABGobj: ',num2str(FvalABG(iter_out))]);
    end
    % Check the convergence condition
    if   iter_out>1 && abs( FvalABG(iter_out) - FvalABG(iter_out-1) ) < absTol * FvalABG(iter_out-1)
        break;
    end
end

if dbFlag
    disp(['MTLL converged with lambda1:',num2str(lambda1),...
        ' lambda2:',num2str(lambda2),...
        ' lambda3:',num2str(lambda3),...
        ' numK:',num2str(numK),...
        ', at the ',num2str(iter_out),'-th iteration with value: ',num2str(FvalABG(iter_out))]);
end

%% Save results
theta = cell(numT,1);
for t = 1:numT
    theta{t} = (A+B{t})*(G{t}(:,1)+G{t}(:,2:numN(t)+1));
end
STATS.Fval = FvalABG;
end

%% Warm Up by ridge regression
function W = warmUp(data, target)
    numT = length(data);
    numD = size(data{1},1);
    W = zeros(numD,numT); 
    lambda = 1e-2;
    for t= 1:numT
        X = (data{t})';
        Y = target{t};
        W(:,t) = (X'*X+lambda*eye(numD))\(X'*Y); % sove w by ridge regression
    end
end

%% Calculate the loss function
function loss = calLoss(Z,Y,A,B,G,M)
theta = (A+B) * (G*M);
Ta = theta(:);
loss = sum((Y-Z*Ta).^2);
end

%% Calculate the gradient w.r.t. G
function [grad_G,LL] = calGradG(Z,Y,A,B,G_hat,lambda2,S,L,M,N)
numN = size(Y,1);
numK = size(G_hat,1);
C = A+B;
temp1 = Z*kron(M',C);
vecGrad_G1 = (-2)*temp1'*(Y-temp1*G_hat(:));
Ik = speye(numK); 
D = diag(sum(S.*L,2));
W = S.*L;
LL = D-W;
temp2 = kron(N',Ik);
temp3 = G_hat*N*LL;
vecGrad_G2 = (4*lambda2)*temp2'*temp3(:);
vecGrad_G = vecGrad_G1 + vecGrad_G2;
grad_G = reshape(vecGrad_G,[numK,numN+1]);
end

%% Calculate the objective value w.r.t G
function fval = calFvalG(Z,Y,A,B,G,LL,M,N,lambda2,lambda3)
Gn = G*N;
loss = calLoss(Z,Y,A,B,G,M);
R2 = trace(Gn*LL*Gn');
R3 = sum(abs(G(:)));
fval = loss + 2*lambda2*R2 +lambda3*R3;
end

%% Calculate the objective value w.r.t G and L
function fval = calFvalGL(Z,Y,A,B,G,S,L,M,lambda2,lambda3)
numN = size(Y,1);
loss =  calLoss(Z,Y,A,B,G,M);
Gtemp = (G(:,2:(numN+1)))';
dist2 =  pdist2(Gtemp,Gtemp,'squaredeuclidean');
R2 = sum(S.*(L.*dist2+(1/4).*(1./L)),'all');
R3 = sum(abs(G(:)));
fval = loss + lambda2*R2+ lambda3*R3;
end

%% Calculate the objective value w.r.t A,B and G
function fval = calFvalABG(Z,Y,A,B,G,S,M,lambda2,lambda3,lambda1)
numT = length(Y);
loss = 0;
R11 = 0;
R12 = 0;
R2 = 0;
R3 = 0;
for t=1:numT
    loss = loss + calLoss(Z{t},Y{t},A,B{t},G{t},M{t});
    numN = length(Y{t});
    Gtemp = (G{t}(:,2:(numN+1)))';
    dist =  pdist2(Gtemp,Gtemp);
    R2 = R2 + sum(S{t}.*dist,'all');
    R3 = R3 + sum(abs(G{t}(:)));
    R11 = R11 + sum(B{t}(:).^2);
end
R12 = sum(A(:).^2);
fval = loss + lambda1*(R11 + R12) + lambda2*R2 + lambda3*R3;
end

%% The proximal operator of the l1 norm
function x = prox_l1(v, lambda)
    x = sign(v).*max(abs(v) - lambda,0);
end
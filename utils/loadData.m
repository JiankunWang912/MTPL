function [X_train, Y_train, X_validation, Y_validation, X_test, Y_test] = loadData(dataset)
load_data = load(dataset);
numT = length(load_data.trSet.X);
X_train = cell(numT,1);
Y_train = cell(numT,1);
X_validation = cell(numT,1);
Y_validation = cell(numT,1);
X_test = cell(numT,1);
Y_test = cell(numT,1);
% add bias
for t = 1:numT
    %X_train{t} = (load_data.trSet.X{t})';
    X_train{t} = ([ load_data.trSet.X{t} ones(size(load_data.trSet.X{t},1),1) ])';
    Y_train{t} = (load_data.trSet.Y{t})';
    %X_validation{t} = (load_data.vaSet.X{t})';
    X_validation{t} = ([ load_data.vaSet.X{t} ones(size(load_data.vaSet.X{t},1),1) ])';
    Y_validation{t} = (load_data.vaSet.Y{t})';
    %X_test{t} = (load_data.teSet.X{t})';
    X_test{t} = ([ load_data.teSet.X{t} ones(size(load_data.teSet.X{t},1),1) ])';
    Y_test{t} = (load_data.teSet.Y{t})';
end
end
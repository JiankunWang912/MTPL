function [sourceNodes, targetNodes, weights] = knnIndexToGraphEdges(neighbors, distances)
nNodes = size(neighbors,1);
k = [];
for i = 1:nNodes
    k = [ k size(neighbors{i},2)];
end
sourceNodes = repelem(1:nNodes, k);
temp = cellfun(@transpose,neighbors,'UniformOutput',false);
targetNodes =(cell2mat(temp))';
ttemp = cellfun(@transpose,distances,'UniformOutput',false);
weights =(cell2mat(ttemp))';
nonSelfEdges = find(targetNodes - sourceNodes);
% remove self-edges
sourceNodes = sourceNodes(nonSelfEdges);
targetNodes = targetNodes(nonSelfEdges);
weights = weights(nonSelfEdges);
end
function final_Str = buildSimilarityGraph(final_Xtr, numNeighbor)
numT = length(final_Xtr);
final_Str = cell(numT,1);
for t = 1:numT
    numN = size(final_Xtr{t},2);
    [neighbors, distances] = knnsearch(final_Xtr{t}', final_Xtr{t}', 'K', numNeighbor + 1, 'IncludeTies',true);
    [sourceNodes, targetNodes, weights] = knnIndexToGraphEdges(neighbors, distances);
    G = digraph(sourceNodes, targetNodes, weights);
    edge=table2array(G.Edges);
    edgeIndex = edge(:,1:2);
    edgeDistance = edge(:,3);
    edgeWeight = exp(-(0.01).*edgeDistance);
    B = mat2cell(edgeIndex, size(edgeIndex, 1), ones(1, 2));
    final_Str{t}=zeros(numN,numN);
    final_Str{t}(sub2ind(size(final_Str{t}), B{:})) = edgeWeight;
    final_Str{t} = sparse((final_Str{t}+final_Str{t}')/2);
end
end
function initial_lengthscale(X)
    if size(X,1) > 10000
        D = pairwise(SqEuclidean(),X[sample(1:size(X,1),10000,replace=false),:],dims=1)
    else
        D = pairwise(SqEuclidean(),X,dims=1)
    end
    return sqrt(mean([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)]))
end

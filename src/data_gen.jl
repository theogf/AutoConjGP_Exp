function randomf(X)
    return X[:,1].^2+cos.(X[:,1]).*sin.(X[:,2])-sin.(X[:,1]).*tanh.(X[:,2])-X[:,1]
    # return X[:,1].*sin.(X[:,2])
end


function sample_gaussian_process(X,kernel,noise)
    N = size(X,1)
    K = kernelmatrix(X,kernel)+noise*Diagonal{Float64}(I,N)
    return rand(MvNormal(zeros(N),K))
end

function generate_uniform_data(N,dim,box_width)
    X = rand(N,dim)*box_width
end


function generate_grid_data(N,dim,box_width)
    if dim == 1
        return reshape(collect(range(0,box_width,length=N)),N,1)
    else
        x1 = range(0,box_width,length=N)
        x2 = range(0,box_width,length=N)
        return hcat([i for i in x1, j in x2][:],[j for i in x1, j in x2][:])
    end
end

function generate_gaussian_data(N,dim,variance=1.0)
    if dim == 1
        d = Normal(0,sqrt(variance))
        X = rand(d,N,1)
    else
        d = MvNormal(zeros(dim),sqrt(variance)*Diagonal{Float64}(I,dim))
        X = rand(d,N)
    end
end


function generate_random_walk_data(N,dim,lambda)
    d = Poisson(lambda)
    i = Vector()
    # if dim == 1
        push!(i,1)
        i_t = 1
        while i_t < N
            i_t = min(N,i_t+rand(d)+1)
            push!(i,i_t)
        end
    # else
        # push!(i,CartesianIndex(1,1))
        # i_t1 = 1; i_t2 = 1
        # while i_t1 < N && i_t2 < N
            # i_t1 = min(N,i_t1+rand(d))
            # i_t2 = min(N,i_t2+rand(d))
            # push!(i,CartesianIndex(i_t1,i_t2))
        # end
    # end
    return i
end

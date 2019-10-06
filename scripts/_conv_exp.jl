using DrWatson;
quickactivate(joinpath(@__DIR__,".."))
include(srcdir("intro.jl"))
using CSV, AugmentedGaussianProcesses
using PyCall
gpflow = pyimport("gpflow")
tf = pyimport("tensorflow")
@show= ARGS[1]
try
    dict = load(f)
    problem_type = Dict("covtype"=>:classification,"heart"=>:classification,"HIGGS"=>:classification,"SUSY"=>:classification,"CASP"=>:regression,"airline"=>:regression)

    likelihood_GD = Dict(:Logistic=>py"BernoulliLogit()",:Matern32=>py"Matern32()",
        :Laplace=>py"Laplace()",:StudentT=>py"gpflow.likelihoods.StudentT(3.0)")
    likelihood2problem = Dict("BernoulliLogit"=>:classification,"Matern32"=>:regression,"Matern32"=>:regression,"StudentT"=>:regression,"Laplace"=>:regression)
    likelihood_CAVI = Dict(:Logistic=>LogisticLikelihood(),:Matern32=>GenMatern32Likelihood(),
        :Laplace=>LaplaceLikelihood(),:StudentT=>StudentTLikelihood(3.0))
    iter_points= vcat(1:9,10:5:99,100:50:999,1e3:1e3:(1e4-1),1e4:1e4:1e5)
    file_name = dict[:file_name]
    problem = problem_type[file_name]
    base_file = datadir("datasets",string(problem),"large",file_name)

    ## Load and preprocess the data
    global data = isfile(base_file*".h5") ? h5read(base_file*".h5","data") : Matrix(CSV.read(base_file*".csv",header=false))
    X = data[:,2:end]; y = data[:,1]; rescale!(X,obsdim=1);
    (N,nDim) = size(X)
    if problem == :classification
        ys = unique(y)
        @assert length(ys) == 2
        y[y.==ys[1]] .= 1; y[y.==ys[2]] .= -1
    elseif problem == :regression
        rescale!(y,obsdim=1)
    end
    l = initial_lengthscale(X)
    @info "Loaded data and initialized lengthscale : $l"
    CAVI_kernel = RBFKernel(l,variance=1.0,dim=nDim)
    ## Get simulation parameters
    doCAVI = dict[:doCAVI]
    doGD = dict[:doGD]
    doNGD = dict[:doNGD]
    nIter = dict[:conv_max]
    tMax = dict[:time_max]
    likelihood = dict[:likelihood]
    nMinibatch = dict[:nMinibatch]
    nInducing = dict[:nInducing]
    Z = AugmentedGaussianProcesses.KMeansInducingPoints(X,nInducing,nMarkov=10)
    N_test_max = 10000
    @info "Created inducing points location matrix"
    params = @ntuple(likelihood,nMinibatch,nInducing,nIter)
    for ((X_train,y_train),(X_test,y_test)) in kfolds((X,y),10,obsdim=1)
        if length(y_test) > N_test_max
            subset = sample(1:length(y_test),N_test_max,replace=false)
            X_test = X_test[subset,:]
            y_test = y_test[subset]
        end
        ## Run Augmented
        if doCAVI
            @info "Training CAVI"
            LogArrays = []
            model = SVGP(X_train,y_train,CAVI_kernel,likelihood_CAVI[likelihood],AnalyticSVI(nMinibatch),nInducing)
            train!(model,iterations=5)
            @info "Finished pre-training"
                model = SVGP(X_train,y_train,CAVI_kernel,likelihood_CAVI[likelihood],AnalyticSVI(nMinibatch),nInducing)
            model.Z[1] = Z
            t_first = time_ns()
            try
                train!(model,iterations=nIter,callback=cbcavi(X_test,y_test,LogArrays))
            catch  e
                if !isa(e,InterruptException)
                    rethrow(e)
                end
            end
            training_df = DataFrame(hcat(LogArrays...)',[:i,:t_init,:metric,:nll,:ELBO,:t_end])
            results = DataFrame([[:CAVI],[likelihood],[nInducing],[nMinibatch],[t_first],[training_df]],[:model,:likelihood,:nInducing,:nMinibatch,:t_first,:training_df])

            @tagsave(datadir("part_3",file_name,savename("CAVI",params,"bson")),@dict results)
        end
        ## Run NGD
        if doNGD
            @info "Training NGD"
            LogArrays = []
            GD_kernel = gpflow.kernels.RBF(nDim,lengthscales=l,ARD=true)
            global model = gpflow.models.SVGP(X_train, Float64.(reshape(y_train,(length(y_train),1))),kern=deepcopy(GD_kernel),likelihood=likelihood_GD[likelihood],num_latent=1,Z=Z,minibatch_size=nMinibatch)
            t_first = time_ns()
            try
                run_nat_grads_with_adam(model,nIter,X_test,y_test,LogArrays,callback=cbgd,time_max=tMax,kernel_fixed=false)
            catch e
                if !isa(e,InterruptException)
                    rethrow(e)
                end
            end
            training_df = DataFrame(hcat(LogArrays...)',[:i,:t_init,:metric,:nll,:ELBO,:t_end])
            results = DataFrame([[:NGD],[likelihood],[nInducing],[nMinibatch],[t_first],[training_df]],[:model,:likelihood,:nInducing,:nMinibatch,:t_first,:training_df])
            @tagsave(datadir("part_3",file_name,savename("NGD",params,"bson")),@dict results
                    )
        end
        ## Run GD
        if doGD
            @info "Training GD"
            LogArrays = []
            GD_kernel = gpflow.kernels.RBF(nDim,lengthscales=l,ARD=true)
            model = gpflow.models.SVGP(X_train, Float64.(reshape(y_train,(length(y_train),1))),kern=deepcopy(GD_kernel),likelihood=likelihood_GD[likelihood],num_latent=1,Z=Z,minibatch_size=nMinibatch)
            t_first = time_ns()
            try
                run_grads_with_adam(model,nIter,X_test,y_test,LogArrays,callback=cbgd,time_max=tMax,kernel_fixed=false)
            catch e
                if !isa(e,InterruptException)
                    rethrow(e)
                end
            end
            training_df = DataFrame(hcat(LogArrays...)',[:i,:t_init,:metric,:nll,:ELBO,:t_end])
            results = DataFrame([[:GD],[likelihood],[nInducing],[nMinibatch],[t_first],[training_df]],[:model,:likelihood,:nInducing,:nMinibatch,:t_first,:training_df])
            @tagsave(datadir("part_3",file_name,savename("GD",params,"bson")),@dict results)
        end
        break;
    end
catch e
    rm(f)
    rethrow(e)
end
rm(f)

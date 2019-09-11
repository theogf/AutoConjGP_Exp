using AugmentedGaussianProcesses
using MLDataUtils, CSV
using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(joinpath(srcdir(),"intro.jl"))
defaultdic = Dict(:nSamples=>1000,:nIterA=>200,:nIterVI=>500,
                    :filename=>"housing.csv",:kernel=>RBFKernel,
                    :likelihood=>LaplaceLikelihood(1.0),
                    :kfold=>10,:nfold=>2,:GS=>true,:AVI=>true,:VI=>true)
function run_vi_exp(dict=defaultdic)
    ## Parameters and data
    nSamples = dict[:nSamples];
    nIterA = dict[:nIterA]
    nIterVI = dict[:nIterVI];
    filename = dict[:filename]
    data = Matrix(CSV.read(joinpath(datadir(),"exp_raw/",filename),header=false))
    y = data[:,1]; X = data[:,2:end]; (N,nDim) = size(X)
    @info "Data loaded from file $filename"
    l = initial_lengthscale(X)
    kerneltype = dict[:kernel]
    kernel = kerneltype(l)
    burnin=1
    likelihood = dict[:likelihood]
    kfold = dict[:kfold]
    nfold = dict[:nfold]
    doGS = dict[:GS]
    doAVI = dict[:AVI]
    doVI = dict[:VI]
    i=1
    for ((X_train,y_train),(X_test,y_test)) in kfolds((X,y),obsdim=1,k=kfold)
        ## Computing truth
        if doGS
            @info "Starting sampling via Gibbs sampler"
            truemodel = VGP(X_train,y_train,kernel,likelihood,GibbsSampling(samplefrequency=1,nBurnin=0),verbose=0,optimizer=false)
            train!(truemodel,iterations=nSamples+1)
        end
        ## Computing the augmented model
        if doAVI
            @info "Starting training of augmented model"
            amodel = VGP(X_train,y_train,kernel,likelihood,AnalyticVI(),verbose=0,optimizer=false)
            train!(amodel,iterations=nIterA)
        end
        ## Computing the classical model
        if doVI
            try
                @info "Starting training of classic model"
                vimodel = VGP(X_train,y_train,kernel,likelihood,QuadratureVI(),verbose=0,optimizer=false)
                train!(vimodel,iterations=nIterVI)
            catch e
                continue #TODO treat error
            end
        end
        i+=1
        if i > nfold
            break;
        end
    end
end

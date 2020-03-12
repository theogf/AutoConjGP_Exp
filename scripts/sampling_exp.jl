using DrWatson
quickactivate(@__DIR__,"AutoConjugate")
include(srcdir("intro.jl"))
using Turing, MCMCChains
using AugmentedGaussianProcesses
using MLDataUtils, CSV, StatsFuns, HDF5
Turing.setadbackend(:reverse_diff)

### Defining the Turing models for HMC and Gibbs

@model laplacemodel(x,y,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ Laplace(f[i],β_l)
    end
end

@model logisticmodel(x,y,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ Bernoulli(logistic(f[i]))
    end
end

@model studenttmodel(x,y,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ LocationScale(f[i],one(f[i]),TDist(ν))
    end
end

struct Matern32Turing{T} <: ContinuousUnivariateDistribution
    μ::T
end

Distributions.logpdf(d::Matern32Turing,x::Real) = log(one(x)+sqrt(3*abs2(x-d.μ)))-sqrt(3*abs2(x-d.μ))

@model matern32model(x,y,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ Matern32Turing(f[i])
    end
end

### Create a dictionary of parameters to work with dict_list creates an array of dictionaries

defaultdictsamp = Dict(:nChains=>5,:nSamples=>10000,:file_name=>"housing",
                    :likelihood=>:StudentT,
                    :doHMC=>!true,:doMH=>!true,:doNUTS=>!true,:doGibbs=>!true)
dictlistsamp = Dict(:nChains=>5,:nSamples=>10000,:file_name=>"housing",
                    :likelihood=>:Matern32,:epsilon=>[0.1],:n_steps=>[10],
                    :doHMC=>!true,:doMH=>true,:doNUTS=>!true,:doGibbs=>true)
dictlistsamp = dict_list(dictlistsamp)

ν = 3.0
β_l = 1.0
## Mapping to use the same likelihood in all models
likelihood2gibbs = Dict(:Logistic=>LogisticLikelihood(),:Laplace=>LaplaceLikelihood(β_l),:StudentT=>StudentTLikelihood(ν),:Matern32=>Matern3_2Likelihood())
likelihood2turing = Dict(:Logistic=>logisticmodel,:Laplace=>laplacemodel,:StudentT=>studenttmodel,:Matern32=>matern32model)
likelihood2ptype = Dict(:Logistic=>:classification,:Laplace=>:regression,:StudentT=>:regression,:Matern32=>:regression)

## Parameters and data
function sample_exp(dict=defaultdictsamp)
    nSamples = dict[:nSamples]; # Number of samples to take
    nChains = dict[:nChains]; # Number of chains
    file_name = dict[:file_name] # Name of the dataset to load
    doGibbs = dict[:doGibbs] # Flag for running Gibbs sampling
    doHMC = dict[:doHMC] # Flag for running HMC
    doNUTS = dict[:doNUTS] # Flag for running NUTS
    doMH = dict[:doMH] # Flag for running Metropolis Hasting
    likelihood = dict[:likelihood] # Likelihood name
    base_file = datadir("datasets",string(likelihood2ptype[likelihood]),"small",file_name)
    ## Load and preprocess the data
    data = isfile(base_file*".h5") ? h5read(base_file*".h5","data") : Matrix(CSV.read(base_file*".csv",header=false))
    y = data[:,1]; X = data[:,2:end]; (N,nDim) = size(X)
    rescale!(X,obsdim=1)
    l = initial_lengthscale(X)
    # Readapt the target given the tool used
    y_turing = []
    if likelihood2ptype[likelihood] == :classification
        ys = unique(y)
        @assert length(ys) == 2
        y[y.==ys[1]].= 1; y[y.==ys[2]].= -1
        y_turing = Int64.((y.+1.0)./2)
        global y_gp = Vector(Bool.(y_turing))
    elseif likelihood2ptype[likelihood] == :regression
        rescale!(y,obsdim=1);
        y_turing = copy(y)
        y_gp = copy(y)
    end
    # Create a kernel and the kernel matrix with the cholesky decomposition
    kernel = transform(SqExponentialKernel(),1/l)
    K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+1e-4*I
    L = Matrix(cholesky(K).L)
    burnin=1
    β = 1.0
    # Empty containers for chains and times
    all_chains = []
    times = []
    params_samp = @ntuple(nSamples)
    ## Training Gibbs Sampling
    if doGibbs
        GSsamples = zeros(nSamples,N,nChains)
        @info "Starting chains of Gibbs sampler"
        times_Gibbs = []
        for i in 1:nChains
            @info "Gibbs chain $i/$nChains"
            amodel = VGP(X,y,kernel,likelihood2gibbs[likelihood],GibbsSampling(samplefrequency=1,nBurnin=0),verbose=2,optimizer=false)
            t = @elapsed train!(amodel,iterations=nSamples+1)
            GSsamples[:,:,i] = transpose(hcat(amodel.inference.sample_store[1]...))
            push!(times_Gibbs,t)
        end
        chains = DataFrame(Chains(GSsamples,string.("f[",1:N,"]")))
        infos = DataFrame(reshape(["GS",nSamples,mean(times_Gibbs),0,0,0,0],1,:),[:alg,:nSamples,:time,:epsilon,:n_step,:n_adapt,:accept])
        @tagsave(datadir("part_1",string(likelihood),savename("GS",params_samp,"bson")), @dict chains infos)
        # write(datadir("part_1",string(likelihood),"Gibbs.jls"),GSchain)
        push!(all_chains,chains)
        push!(times,times_Gibbs)
    end
    ## Training HMC
    if doHMC
        @info "Starting chains of HMC"
        times_HMC = []
        HMCchains = []
        epsilon = dict[:epsilon]; n_step = dict[:n_steps]
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed HMCchain = sample(likelihood2turing[likelihood](X,y_turing,L), HMC(epsilon,n_step),nSamples)
            push!(HMCchains,HMCchain)
            push!(times_HMC,t)
        end
        chains = DataFrame(chainscat(HMCchains...))
        infos = DataFrame(reshape(["HMC",nSamples,mean(times_HMC),epsilon,n_step,0,0],1,:),[:alg,:nSamples,:time,:epsilon,:n_step,:n_adapt,:accept])
        params_HMC = @ntuple(nSamples,epsilon,n_step)
        @tagsave(datadir("part_1",string(likelihood),savename("HMC",params_HMC,"bson")), @dict chains infos)
        push!(all_chains,chains)
        push!(times,times_HMC)
    end
    ## Training NUTS
    if doNUTS
        @info "Starting chains of NUTS"
        times_NUTS = []
        NUTSchains = []
        n_adapt = 100;
        accept = 0.2
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed NUTSchain = sample(likelihood2turing[likelihood](X,y_turing,L), NUTS(nSamples,n_adapt,accept))
            push!(NUTSchains,NUTSchain)
            push!(times_NUTS,t)
        end
        chains = DataFrame(chainscat(NUTSchains...))
        infos = DataFrame(reshape(["NUTS",nSamples,mean(times_NUTS),0,0,n_adapt,accept],1,:),[:alg,:nSamples,:time,:epsilon,:n_step,:n_adapt,:accept])
        params_NUTS = @ntuple(nSamples,n_adapt,accept)
        @tagsave(datadir("part_1",string(likelihood),savename("NUTS",params_NUTS,"bson")), @dict chains infos)
        push!(all_chains,chains)
        push!(times,times_NUTS)
    end
    ## Training MH
    if doMH
        @info "Starting chains of MH"
        times_MH = []
        MHchains = []
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed MHchain = sample(likelihood2turing[likelihood](X,y_turing,L), MH(nSamples))
            push!(MHchains,MHchain)
            push!(times_MH,t)
        end
        chains = DataFrame(chainscat(MHchains...))
        infos = DataFrame(reshape(["MH",nSamples,mean(times_MH),0,0,0,0],1,:),[:alg,:nSamples,:time,:epsilon,:n_step,:n_adapt,:accept])
        @tagsave(datadir("part_1",string(likelihood),savename("MH",params_samp,"bson")), @dict chains infos)
        push!(all_chains,chains)
        push!(times,times_MH)
    end
    return all_chains,times
end

# Run over the set of dictionnaries
map(sample_exp,dictlistsamp)

# Run over one dictionary
sample_exp()

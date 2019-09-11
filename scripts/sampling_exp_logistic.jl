using Turing, MCMCChains
using AugmentedGaussianProcesses
using MLDataUtils, CSV
using DrWatson
quickactivate("/home/theo/experiments/AutoConj")
include(joinpath(srcdir(),"intro.jl"))
Turing.setadbackend(:reverse_diff)

@model logisticmodel(x,y,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ Bernoulli(logistic(f[i]))
    end
end

defaultdictsamplogistic = Dict(:nChains=>5,:nSamples=>2000,:file_name=>".csv",
                    :doTuring=>true,:doGibbs=>true)
## Parameters and data
function sample_exp_laplace(dict=defaultdictsamplogistic)
    nSamples = dict[:nSamples];
    nChains = dict[:nChains];
    file_name = dict[:file_name]
    doTuring = dict[:doTuring]
    doGibbs = dict[:doGibbs]
    data = Matrix(CSV.read(joinpath(datadir(),"exp_raw",file_name),header=false))
    y = data[:,1]; X = data[:,2:end]; (N,nDim) = size(X)
    rescale!(X,obsdim=1)
    l = initial_lengthscale(X)

    kernel = RBFKernel(l)
    K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+1e-4*I
    L = Matrix(cholesky(K).L)
    burnin=1
    β = 1.0
    chains = []
    times = []
    ## Training Turing
    if doTuring
        @info "Starting chains of HMC"
        times_NUTS = []
        NUTSchains = []
        # NUTSchain = mapreduce(c->sample(laplacemodel(X,y,β,L),NUTS(nSamples,100,0.6),progress=true),chainscat,1:nChains)
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed NUTSchain = sample(logisticmodel(X,y,β,L), HMC(nSamples,0.36,2))
            # t = @elapsed NUTSchain = sample(laplacemodel(X,y,β,L), NUTS(nSamples,100,0.6))
            push!(NUTSchains,NUTSchain)
            push!(times_NUTS,t)
        end
        NUTSchains = chainscat(NUTSchains...)
        push!(chains,NUTSchains)
        push!(times,times_NUTS)
    end
    ## Training Gibbs Sampling
    if doGibbs
        GSsamples = zeros(nSamples,N,nChains)
        @info "Starting chains of Gibbs sampler"
        times_Gibbs = []
        for i in 1:nChains
            @info "Gibbs chain $i/$nChains"
            amodel = VGP(X,y,kernel,LogisticLikelihood(),GibbsSampling(samplefrequency=1,nBurnin=0),verbose=0,optimizer=false)
            t = @elapsed train!(amodel,iterations=nSamples+1)
            GSsamples[:,:,i] = transpose(hcat(amodel.inference.sample_store[1]...))
            push!(times_Gibbs,t)
        end
        GSchain = Chains(GSsamples,string.("f[",1:N,"]"))
        push!(chains,GSchain)
        push!(times,times_Gibbs)
    end
    return chains,times
end

chains, times = sample_exp_logistic()

function treat_chain(chain)
    burnin = mean(mean.(getindex.(heideldiag(chain),Symbol("Burn-in"))))
    gelman = mean(gelmandiag(chain)[:PSRF])
    acorlag1 = mean(mean.(abs,getindex.(autocor(chain),Symbol("lag 1"))))
    return burnin, gelman, acorlag1
end
treat_chain.(chains)
mean(mean.(getindex.(heideldiag(NUTSchain),Symbol("Burn-in"))))
## Diagnostics
mean(mean.(abs,getindex.(autocor(NUTSchain),Symbol("lag 1"))))
println("Gelman: NUTS : $(mean(gelmandiag(NUTSchain)[:PSRF])), GibbsSampling : $(mean(gelmandiag(GSchain)[:PSRF]))")
println("Heidel: NUTS : $(mean(mean.(getindex.(heideldiag(NUTSchain),Symbol("Burn-in"))))), GibbsSampling : $(mean(mean.(getindex.(heideldiag(GSchain),Symbol("Burn-in")))))")

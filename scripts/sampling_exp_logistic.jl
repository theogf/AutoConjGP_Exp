using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using Turing, MCMCChains
using AugmentedGaussianProcesses
using MLDataUtils, CSV
using StatsFuns
Turing.setadbackend(:reverse_diff)

@model logisticmodel(x,y,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ Bernoulli(logistic(f[i]))
    end
end

defaultdictsamplogistic = Dict(:nChains=>1,:nSamples=>100,:file_name=>"heart.csv",
                    :doHMC=>true,:doMH=>true,:doNUTS=>true,:doGibbs=>true)
## Parameters and data
function sample_exp_logistic(dict=defaultdictsamplogistic)
    nSamples = dict[:nSamples];
    nChains = dict[:nChains];
    file_name = dict[:file_name]
    doGibbs = dict[:doGibbs]
    doHMC = dict[:doHMC]
    doNUTS = dict[:doNUTS]
    doMH = dict[:doMH]
    data = Matrix(CSV.read(joinpath(datadir(),"datasets","classification","small",file_name),header=false))
    y = data[:,1]; X = data[:,2:end]; (N,nDim) = size(X)
    y_turing = (y.+1.0)./2
    rescale!(X,obsdim=1)
    l = initial_lengthscale(X)

    kernel = RBFKernel(l)
    K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+1e-4*I
    L = Matrix(cholesky(K).L)
    burnin=1
    !isdir(datadir("part_1","logistic")) ? mkdir(datadir("part_1","logistic")) : nothing
    chains = []
    times = []
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
        global GSchain = Chains(GSsamples,string.("f[",1:N,"]"))
        write(datadir("part_1","logistic","Gibbs.jls"),GSchain)
        push!(chains,GSchain)
        push!(times,times_Gibbs)
    end
    ## Training HMC
    if doHMC
        @info "Starting chains of HMC"
        times_HMC = []
        HMCchains = []
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed HMCchain = sample(logisticmodel(X,y_turing,L), HMC(0.36,2), nSamples)
            push!(HMCchains,HMCchain)
            push!(times_HMC,t)
        end
        HMCchains = chainscat(HMCchains...)
        write(datadir("part_1","logistic","HMC.jls"),HMCchains)
        push!(chains,HMCchains)
        push!(times,times_HMC)
    end
    ## Training NUTS
    if doNUTS
        @info "Starting chains of NUTS"
        times_NUTS = []
        NUTSchains = []
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed NUTSchain = sample(logisticmodel(X,y_turing,L), NUTS(100,0.2), nSamples+100)
            push!(NUTSchains,NUTSchain)
            push!(times_NUTS,t)
        end
        NUTSchains = chainscat(NUTSchains...)
        write(datadir("part_1","logistic","NUTS.jls"),NUTSchains)
        push!(chains,NUTSchains)
        push!(times,times_NUTS)
    end
    ## Training MH
    if doMH
        @info "Starting chains of MH"
        times_MH = []
        MHchains = []
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed MHchain = sample(logisticmodel(X,y_turing,L), MH(),nSamples)
            push!(MHchains,MHchain)
            push!(times_MH,t)
        end
        MHchains = chainscat(MHchains...)
        write(datadir("part_1","logistic","MH.jls"),MHchains)

        push!(chains,MHchains)
        push!(times,times_MH)
    end
    return chains,times
end

chains, times = sample_exp_logistic()

function treat_chain(chain)
    gelman = mean(gelmandiag(chain)[:PSRF]) |> display
    acorlag1 = mean(mean.(abs,getindex.(autocor(chain),Symbol("lag 1"))))|> display
    burnin = mean(mean.(getindex.(heideldiag(chain),Symbol("Burn-in")))) |> display
    return burnin, gelman, acorlag1
end
treat_chain.(chains)
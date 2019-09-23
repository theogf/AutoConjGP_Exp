using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using Turing, MCMCChains
using AugmentedGaussianProcesses
using MLDataUtils, CSV
Turing.setadbackend(:reverse_diff)

@model laplacemodel(x,y,β,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ Laplace(f[i],β)
    end
end

defaultdictsamplaplace = Dict(:nChains=>1,:nSamples=>100,:file_name=>"housing.csv",
                    :doHMC=>true,:doMH=>true,:doNUTS=>true,:doGibbs=>true)
## Parameters and data
function sample_exp_laplace(dict=defaultdictsamplaplace)
    nSamples = dict[:nSamples];
    nChains = dict[:nChains];
    file_name = dict[:file_name]
    doGibbs = dict[:doGibbs]
    doHMC = dict[:doHMC]
    doNUTS = dict[:doNUTS]
    doMH = dict[:doMH]
    data = Matrix(CSV.read(joinpath(datadir(),"datasets","regression","small",file_name),header=false))
    y = data[:,1]; X = data[:,2:end]; (N,nDim) = size(X)
    rescale!(y,obsdim=1); rescale!(X,obsdim=1)
    l = initial_lengthscale(X)

    kernel = RBFKernel(l)
    K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+1e-4*I
    L = Matrix(cholesky(K).L)
    burnin=1
    β = 1.0
    !isdir(datadir("part_1","laplace")) ? mkdir(datadir("part_1","laplace")) : nothing
    chains = []
    times = []
    ## Training Gibbs Sampling
    if doGibbs
        GSsamples = zeros(nSamples,N,nChains)
        @info "Starting chains of Gibbs sampler"
        times_Gibbs = []
        for i in 1:nChains
            @info "Gibbs chain $i/$nChains"
            amodel = VGP(X,y,kernel,LaplaceLikelihood(β),GibbsSampling(samplefrequency=1,nBurnin=0),verbose=0,optimizer=false)
            t = @elapsed train!(amodel,iterations=nSamples+1)
            GSsamples[:,:,i] = transpose(hcat(amodel.inference.sample_store[1]...))
            push!(times_Gibbs,t)
        end
        global GSchain = Chains(GSsamples,string.("f[",1:N,"]"))
        write(datadir("part_1","laplace","Gibbs.jls"),GSchain)
        push!(chains,GSchain)
        push!(times,times_Gibbs)
    end
    ## Training HMC
    if doHMC
        @info "Starting chains of HMC"
        times_HMC = []
        HMCchains = []
        # HMCchain = mapreduce(c->sample(laplacemodel(X,y,β,L),HMC(nSamples,100,0.6),progress=true),chainscat,1:nChains)
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed HMCchain = sample(laplacemodel(X,y,β,L), HMC(0.36,2), nSamples)
            push!(HMCchains,HMCchain)
            push!(times_HMC,t)
        end
        HMCchains = chainscat(HMCchains...)
        write(datadir("part_1","laplace","HMC.jls"),HMCchains)
        push!(chains,HMCchains)
        push!(times,times_HMC)
    end
    ## Training NUTS
    if doNUTS
        @info "Starting chains of NUTS"
        times_NUTS = []
        NUTSchains = []
        # NUTSchain = mapreduce(c->sample(laplacemodel(X,y,β,L),NUTS(nSamples,100,0.6),progress=true),chainscat,1:nChains)
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed NUTSchain = sample(laplacemodel(X,y,β,L), NUTS(100,0.2), nSamples+100)
            push!(NUTSchains,NUTSchain)
            push!(times_NUTS,t)
        end
        NUTSchains = chainscat(NUTSchains...)
        write(datadir("part_1","laplace","NUTS.jls"),NUTSchains)
        push!(chains,NUTSchains)
        push!(times,times_NUTS)
    end
    ## Training MH
    if doMH
        @info "Starting chains of MH"
        times_MH = []
        MHchains = []
        # MHchain = mapreduce(c->sample(laplacemodel(X,y,β,L),MH(nSamples,100,0.6),progress=true),chainscat,1:nChains)
        for i in 1:nChains
            @info "Turing chain $i/$nChains"
            t = @elapsed MHchain = sample(laplacemodel(X,y,β,L), MH(),nSamples)
            push!(MHchains,MHchain)
            push!(times_MH,t)
        end
        MHchains = chainscat(MHchains...)
        write(datadir("part_1","laplace","MH.jls"),MHchains)

        push!(chains,MHchains)
        push!(times,times_MH)
    end
    return chains,times
end

chains, times = sample_exp_laplace()

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

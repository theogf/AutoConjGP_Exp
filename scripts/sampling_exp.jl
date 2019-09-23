using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using Turing, MCMCChains
using AugmentedGaussianProcesses
using MLDataUtils, CSV
Turing.setadbackend(:reverse_diff)

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
        y[i] ~ LocationScale(f[i],1.0,TDist(ν))
    end
end

defaultdictsamp = Dict(:nChains=>1,:nSamples=>100,:file_name=>"heart.csv",
                    :likelihood=>:Logistic,
                    :doHMC=>true,:doMH=>true,:doNUTS=>true,:doGibbs=>true)
ν = 3.0
β_l = 1.0
likelihood2gibbs = Dict(:Logistic=>LogisticLikelihood(),:Laplace=>LaplaceLikelihood(β_l),:StudentT=>StudentTLikelihood(ν))
likelihood2turing = Dict(:Logistic=>logisticmodel,:Laplace=>laplacemodel,:StudentT=>studenttmodel)
likelihood2ptype = Dict(:Logistic=>:classification,:Laplace=>:regression,:StudentT=>:regression)

## Parameters and data
function sample_exp(dict=defaultdictsamp)
    nSamples = dict[:nSamples];
    nChains = dict[:nChains];
    file_name = dict[:file_name]
    doGibbs = dict[:doGibbs]
    doHMC = dict[:doHMC]
    doNUTS = dict[:doNUTS]
    doMH = dict[:doMH]
    likelihood = dict[:likelihood]
    data = Matrix(CSV.read(joinpath(datadir(),"datasets",string(likelihood2ptype[likelihood]),"small",file_name),header=false))
    y = data[:,1]; X = data[:,2:end]; (N,nDim) = size(X)
    rescale!(X,obsdim=1)
    l = initial_lengthscale(X)
    y_turing = []
    if likelihood2ptype[likelihood] == :classification
        ys = unique(y)
        @assert length(ys) == 2
        y[y.==ys[1]].= 1; y[y.==ys[2]].= -1
        y_turing = (y.+1.0)./2
    elseif likelihood2ptype[likelihood] == :regression
        rescale!(y,obsdim=1);
        y_turing .= y
    end

    kernel = RBFKernel(l)
    K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+1e-4*I
    L = Matrix(cholesky(K).L)
    burnin=1
    β = 1.0
    !isdir(datadir("part_1",string(likelihood))) ? mkdir(datadir("part_1",string(likelihood))) : nothing
    chains = []
    times = []
    ## Training Gibbs Sampling
    if doGibbs
        GSsamples = zeros(nSamples,N,nChains)
        @info "Starting chains of Gibbs sampler"
        times_Gibbs = []
        for i in 1:nChains
            @info "Gibbs chain $i/$nChains"
            amodel = VGP(X,y,kernel,likelihood2gibbs[likelihood],GibbsSampling(samplefrequency=1,nBurnin=0),verbose=0,optimizer=false)
            t = @elapsed train!(amodel,iterations=nSamples+1)
            GSsamples[:,:,i] = transpose(hcat(amodel.inference.sample_store[1]...))
            push!(times_Gibbs,t)
        end
        global GSchain = Chains(GSsamples,string.("f[",1:N,"]"))
        write(datadir("part_1",string(likelihood),"Gibbs.jls"),GSchain)
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
            t = @elapsed HMCchain = sample(likelihood2turing[likelihood](X,y,L), HMC(0.36,2), nSamples)
            push!(HMCchains,HMCchain)
            push!(times_HMC,t)
        end
        HMCchains = chainscat(HMCchains...)
        write(datadir("part_1",string(likelihood),"HMC.jls"),HMCchains)
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
            t = @elapsed NUTSchain = sample(likelihood2turing[likelihood](X,y,L), NUTS(100,0.2), nSamples+100)
            push!(NUTSchains,NUTSchain)
            push!(times_NUTS,t)
        end
        NUTSchains = chainscat(NUTSchains...)
        write(datadir("part_1",string(likelihood),"NUTS.jls"),NUTSchains)
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
            t = @elapsed MHchain = sample(likelihood2turing[likelihood](X,y,L), MH(),nSamples)
            push!(MHchains,MHchain)
            push!(times_MH,t)
        end
        MHchains = chainscat(MHchains...)
        write(datadir("part_1",string(likelihood),"MH.jls"),MHchains)

        push!(chains,MHchains)
        push!(times,times_MH)
    end
    return chains,times
end

chains, times = sample_exp()

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

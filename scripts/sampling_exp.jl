using Turing, MCMCChains
using AugmentedGaussianProcesses
using MLDataUtils, CSV
using DrWatson
quickactivate("/home/theo/experiments/AutoConj")
include(joinpath(srcdir(),"intro.jl"))

## Parameters and data
nSamples = 100;
nChains = 5;
data = Matrix(CSV.read(joinpath(datadir(),"exp_raw/housing.csv"),header=false))
y = data[:,1]; X = data[:,2:end]; (N,nDim) = size(X)
l = initial_lengthscale(X)

kernel = RBFKernel(l)
K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+1e-4*I
L = Matrix(cholesky(K).L)
burnin=1
β = 1.0

## Training Turing

@model laplacemodel(x,y,β,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ Laplace(f[i],β)
    end
end

@info "Starting chains of NUTS"
NUTSchain = mapreduce(c->sample(laplacemodel(X,y,β,L),NUTS(nSamples,50,0.6),progress=true),chainscat,1:nChains)

## Training Gibbs Sampling

GSsamples = zeros(nSamples,N,nChains)
@show "Starting chains of Gibbs sampler"
Distributed.@distributed for i in 1:nChains
    amodel = VGP(X,y,RBFKernel(0.1),LaplaceLikelihood(β),GibbsSampling(samplefrequency=1,nBurnin=0),verbose=0,optimizer=false)
    train!(amodel,iterations=nSamples+1)
    GSsamples[:,:,i] = transpose(hcat(amodel.inference.sample_store[1]...))
end
GSchain = Chains(GSsamples,string.("f[",1:N,"]"))


## Diagnostics
mean(mean.(getindex.(heideldiag(GSchain),Symbol("Burn-in"))))
mean(mean.(getindex.(heideldiag(NUTSchain),Symbol("Burn-in"))))
mean(mean.(abs,getindex.(autocor(GSchain),Symbol("lag 1"))))
mean(mean.(abs,getindex.(autocor(NUTSchain),Symbol("lag 1"))))
println("Gelman: NUTS : $(mean(gelmandiag(NUTSchain)[:PSRF])), GibbsSampling : $(mean(gelmandiag(GSchain)[:PSRF]))")
println("Gelman: NUTS : $(mean(mean.(getindex.(heideldiag(NUTSchain),Symbol("Burn-in"))))), GibbsSampling : $(mean(mean.(getindex.(heideldiag(GSchain),Symbol("Burn-in")))))")

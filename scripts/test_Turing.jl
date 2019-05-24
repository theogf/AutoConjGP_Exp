using Turing, MCMCChains
using StatsPlots
using Plots
using AugmentedGaussianProcesses
using PDMats, LinearAlgebra




@model basicmodel(x,y,ν,σ,K,L) = begin
    whitef ~ MvNormal(zeros(length(y)),I)
    f = L*whitef
    # shifty = (f-y)/σ
    for i in 1:size(x,1)
        # shifty[i] ~ TDist(ν)
        y[i] ~ LocationScale(f[i],σ,TDist(ν))
    end
end

@model augmodel(x,y,ν,σ,K,f=ones(length(y))) = begin
    ω = Vector{Real}(undef,length(y))
    for i in 1:size(x,1)
        ω[i] ~ InverseGamma(0.5*(ν+1.0),0.5*(σ^2*ν+(y[i]-f[i])^2))
    end
    Σ = Symmetric(inv(Diagonal(inv.(ω))+inv(K)))
    f ~ MvNormal(Σ*(y./ω),Σ)
end

iterations = 1000
N = 100
σ = 1e-1
X = sort(rand(N,1),dims=1)
K = kernelmatrix(X,RBFKernel(0.1))+1e-4*I
L = inv(cholesky(K).L)
y = rand(MvNormal(K+σ*I))
p = scatter(X[:],y,lab="data")
ν = 60.0

augchain2= sample(augmodel(X,y,ν,σ,K),MH(iterations))


chain = sample(basicmodel(X,y,ν,σ,K,L),HMC(iterations,0.1,10))
amodel = VGP(X,y,RBFKernel(0.1),StudentTLikelihood(ν,σ),GibbsSampling(samplefrequency=1,nBurnin=0))
train!(amodel,iterations=iterations+1)
augchain = Chains(reshape(transpose(hcat(amodel.inference.sample_store[1]...)),iterations,:,1),string.("f[",1:N,"]"))
Plots.plot(vec(X),inv(L)*[mean(chain[Symbol("whitef[",i,"]")].value) for i in 1:N],lab="HMC")
Plots.plot!(vec(X),[mean(augchain[Symbol("f[",i,"]")].value) for i in 1:N],lab="Gibbs")
Plots.plot!(vec(X),[mean(augchain2[Symbol("f[",i,"]")].value) for i in 1:N],lab="Gibbs Turing")


##
anim = Animation()

for j in 1:10:iterations
    p = scatter(X[:],y,lab="data",title="i=$j")
    Plots.plot!(vec(X),[mean((chain[Symbol("f[",i,"]")])[1:j].value) for i in 1:N],lab="HMC")
    Plots.plot!(vec(X),[mean((augchain[Symbol("f[",i,"]")])[1:j].value) for i in 1:N],lab="Gibbs")
    Plots.plot!(vec(X),[mean((augchain2[Symbol("f[",i,"]")])[1:j].value) for i in 1:N],lab="Gibbs Turing")
    frame(anim)
end

gif(anim,fps=10)

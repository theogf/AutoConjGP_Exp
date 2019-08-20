using Turing, MCMCChains
using StatsPlots
using Plots
using AugmentedGaussianProcesses
using PDMats, LinearAlgebra




@model basicmodel(x,y,ν,σ,K,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
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
##
iterations = 10000
burnin=1000
N = 100
σ = 1.0
X = sort(rand(N,1),dims=1)
K = kernelmatrix(X,RBFKernel(0.1))+1e-5*I
L = Matrix(cholesky(K).L)
y = rand(MvNormal(K+σ*I))
p = scatter(X[:],y,lab="data")
ν = 3.0

##

augchain2= sample(augmodel(X,y,ν,σ,K),MH(iterations))
Turing.setadbackend(:reverse_diff)
Turing.setadbackend(:forward_diff)

# chain = sample(basicmodel(X,y,ν,σ,K,L),MH(iterations))
chain = sample(basicmodel(X,y,ν,σ,K,L),HMC(iterations,0.1,10))
amodel = VGP(X,y,RBFKernel(0.1),StudentTLikelihood(ν,σ),GibbsSampling(samplefrequency=1,nBurnin=0),verbose=3)
train!(amodel,iterations=iterations+1)
augchain = Chains(reshape(transpose(hcat(amodel.inference.sample_store[1]...)),iterations,:,1),string.("f[",1:N,"]"))
Plots.plot!(vec(X),L*[mean(chain[Symbol("z[",i,"]")].value[burnin:end]) for i in 1:N],lab="HMC")

Plots.plot!(vec(X),[mean(augchain[Symbol("f[",i,"]")].value[burnin:end]) for i in 1:N],lab="Gibbs")
Plots.plot!(vec(X),[mean(augchain2[Symbol("f[",i,"]")].value[burnin:end]) for i in 1:N],lab="Gibbs Turing")
display(p)

##
anim = Animation()

for j in 1:100:iterations
    p = scatter(X[:],y,lab="data",title="i=$j")
    Plots.plot!(vec(X),L*[mean((chain[Symbol("z[",i,"]")])[1:j].value) for i in 1:N],lab="HMC")
    Plots.plot!(vec(X),[mean((augchain[Symbol("f[",i,"]")])[1:j].value) for i in 1:N],lab="Gibbs")
    Plots.plot!(vec(X),[mean((augchain2[Symbol("f[",i,"]")])[1:j].value) for i in 1:N],lab="Gibbs Turing")
    frame(anim)
end

gif(anim,fps=10)
##
NChains = 2


## Run Augmented Version
function augsample(iterations,y,invK,NChains)
    N = length(y)
    augsamples = zeros(iterations,N*2,NChains)
    ω = rand(N)
    f = rand(N)
    @progress for K in 1:NChains
        @progress for i in 1:iterations
            for i in 1:N
                ω[i] = rand(InverseGamma(0.5*(ν+1.0),0.5*(σ^2*ν+(y[i]-f[i])^2)))
            end
            Σ = Symmetric(inv(Diagonal(inv.(ω))+invK))
            f = rand(MvNormal(Σ*(y./ω),Σ))
            augsamples[i,1:N,K] = f
            augsamples[i,(N+1):end,K] = ω
        end
    end
    return augsamples
end
augsamples = augsample(iterations,y,inv(K),NChains)
augchain = Chains(augsamples,vcat(string.("f[",1:N,"]"),string.("omega[",1:N,"]")))
gelmandiag(augchain)
gewekediag(augchain)
heideldiag(augchain)
plot!(X,(mean(augchain).df)[:mean],lab="augchain")
augchain
autocor(augchain)

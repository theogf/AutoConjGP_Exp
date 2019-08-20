using Turing, MCMCChains
using StatsPlots
using Plots
using AugmentedGaussianProcesses
using PDMats, LinearAlgebra




@model basiclaplacemodel(x,y,β,K,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ Laplace(f[i],β)
    end
end
@model auglaplacemodel(x,y,ν,σ,K,f=ones(length(y))) = begin
    ω = Vector{Real}(undef,length(y))
    for i in 1:size(x,1)
        ω[i] ~ GeneralizedIn(1,0.5*(σ^2*ν+(y[i]-f[i])^2))
    end
    Σ = Symmetric(inv(Diagonal(inv.(ω))+inv(K)))
    f ~ MvNormal(Σ*(y./ω),Σ)
end

iterations = 1000
burnin=1000
N = 100
σ = 1.0
X = sort(rand(N,1),dims=1)
K = kernelmatrix(X,RBFKernel(0.1))+1e-4*I
L = Matrix(cholesky(K).L)
y = rand(MvNormal(K+σ*I))
p = scatter(X[:],y,lab="data")
β = 2.0

augchain2= sample(augmodel(X,y,β,K),MH(iterations))
Turing.setadbackend(:reverse_diff)
Turing.setadbackend(:forward_diff)

chain = sample(basicmodel(X,y,β,K,L),MH(iterations))
# chain = sample(basicmodel(X,y,ν,σ,K,L),HMC(iterations,0.1,10))
amodel = VGP(X,y,RBFKernel(0.1),LaplaceLikelihood(β),GibbsSampling(samplefrequency=10,nBurnin=0),verbose=3)
train!(amodel,iterations=iterations+1)
augchain = Chains(reshape(transpose(hcat(amodel.inference.sample_store[1]...)),iterations,:,1),string.("f[",1:N,"]"))
Plots.plot!(vec(X),L*[mean(chain[Symbol("z[",i,"]")].value[burnin:end]) for i in 1:N],lab="HMC")

Plots.plot!(vec(X),[mean(augchain[Symbol("f[",i,"]")].value[burnin:end]) for i in 1:N],lab="Gibbs")
Plots.plot!(vec(X),[mean(augchain2[Symbol("f[",i,"]")].value[burnin:end]) for i in 1:N],lab="Gibbs Turing")
display(p)

y_pred = predict_y(amodel,X)
plot(X,y_pred)

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

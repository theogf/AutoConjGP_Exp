using AugmentedGaussianProcesses
using Plots
using Turing
using MCMCChains, Mamba

iterations = 1000
N = 100
σ = 1e-1
X = sort(rand(N,1),dims=1)
K = kernelmatrix(X,RBFKernel(0.1))+1e-7I
y = rand(MvNormal(K+σ*I))
p = scatter(X[:],y,lab="data")
ν = 3.0


amodel = VGP(X,y,RBFKernel(0.1),StudentTLikelihood(ν),GibbsSampling(samplefrequency=1,nBurnin=0))
train!(amodel,iterations=iterations+1)
augchain = Chains(reshape(transpose(hcat(amodel.inference.sample_store[1]...)),iterations,:,1),string.("f",1:N))

plot(plot(augchain[:f60]),plot(augchain[:f62]))
mean(augchain[:f60].value)
plot(vec(X),y)
plot!(vec(X),[mean(augchain[Symbol("f",i)].value) for i in 1:100])

##

gmodel = Model(
    f = Stochastic(N,
    ()-> MvNormal(zeros(N),K)
    )
    y = Stochastic(1,
    (f)->UnivariateDistribution[LocationScale(f[i],σ,TDist(ν)) for i in 1:N],
    false)
)

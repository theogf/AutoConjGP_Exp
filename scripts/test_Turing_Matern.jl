using Turing, MCMCChains
using StatsPlots
using Plots
using AugmentedGaussianProcesses
using PDMats, LinearAlgebra

@inline logistic(x) = inv.(1.0.+exp.(-x))


@model basicmaternmodel(x,y,ρ,L) = begin
    z ~ MvNormal(zeros(length(y)),I)
    f = L*z
    for i in 1:size(x,1)
        y[i] ~ Bernoulli(logistic(f[i]))
    end
end

iterations = 5000
burnin=1
N = 100
σ = 1.0
X = sort(rand(N,1),dims=1)
kernel = RBFKernel(0.1)
K = kernelmatrix(X,kernel)+1e-6*I
L = Matrix(cholesky(K).L)
true_f = rand(MvNormal(K))
y = sign.(true_f+σ*randn(N))
y_turing = (y.+1.0)./2
p = scatter(X[:],y,lab="data")

Turing.setadbackend(:reverse_diff)
Turing.setadbackend(:forward_diff)

# chain = sample(basiclaplacemodel(X,y,β,L),HMC(iterations,0.1,10))
chain = sample(basiclogisticmodel(X,y_turing,L),NUTS(iterations,200,0.9),progress=true)
amodel = VGP(X,y,RBFKernel(0.1),LaplaceLikelihood(β),GibbsSampling(samplefrequency=1,nBurnin=0),verbose=3,optimizer=false)
train!(amodel,iterations=iterations+1)
augchain = Chains(reshape(transpose(hcat(amodel.inference.sample_store[1]...)),iterations,:,1),string.("f[",1:N,"]"))


##

p = scatter(X[:],y,lab="data")
plot!(X,true_f,lab="True f",color=4)
Mchain = Array(chain)[burnin:end,:]
mchain = L*vec(mean(Mchain,dims=1))
vchain = diag(L*cov(Mchain)*L')
Plots.plot!(vec(X),mchain.+2*sqrt.(vchain),fillrange=mchain.-2*sqrt.(vchain),lw=0.0,lab="",color=1,alpha=0.2)
Plots.plot!(vec(X),L*mean(Mchain,dims=1)',lab="HMC",color=1)

Maugchain_final = Array(augchain)[burnin:end,:]
maugchain_final = vec(mean(Maugchain_final,dims=1))
vaugchain_final = diag(cov(Maugchain_final))
Plots.plot!(vec(X),maugchain_final.+2*sqrt.(vaugchain_final),fillrange=maugchain_final.-2*sqrt.(vaugchain_final),lw=0.0,lab="",color=2,alpha=0.2)
Plots.plot!(vec(X),maugchain_final,lab="Gibbs",color=3)
display(p)

##
anim = Animation()
NNuts = size(chain,1)

@progress for j in burnin:10:iterations
    p = scatter(X[:],y,lab="data")
    Plots.plot!(X,true_f,lab="True f",color=4)
    Plots.plot!(vec(X),maugchain_final,lab="",color=:black,lw=3.0)
    Plots.plot!(vec(X),maugchain_final.+2*sqrt.(vaugchain_final),lw=2.0,linestyle=:dash,lab="",color=:black)
    Plots.plot!(vec(X),maugchain_final.-2*sqrt.(vaugchain_final),lw=2.0,linestyle=:dash,lab="",color=:black)

    Mchain = Array(chain)[burnin:min(j,NNuts),:]
    mchain = L*vec(mean(Mchain,dims=1))
    vchain = diag(L*cov(Mchain)*L')
    Plots.plot!(vec(X),mchain.+2*sqrt.(vchain),fillrange=mchain.-2*sqrt.(vchain),lw=0.0,lab="",color=1,alpha=0.2)
    Plots.plot!(vec(X),L*mean(Mchain,dims=1)',lab="HMC",color=1,lw=2.0)

    Maugchain = Array(augchain)[burnin:j,:]
    maugchain = vec(mean(Maugchain,dims=1))
    vaugchain = diag(cov(Maugchain))
    Plots.plot!(vec(X),maugchain.+2*sqrt.(vaugchain),fillrange=maugchain.-2*sqrt.(vaugchain),lw=0.0,lab="",color=2,alpha=0.2)
    Plots.plot!(vec(X),vec(mean(Maugchain,dims=1)),lab="Gibbs",color=2,title="$j samples",lw=2.0)
    frame(anim)
end

gif(anim,"/home/theo/sampling_evolution.gif",fps=24)


###

function cb(model,iter)
    scatter(X,y,lab="")
    plot(X,model.μ[1],lab="") |> display
end

train!(amodel,iterations=10,callback=cb)

using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Statistics, ForwardDiff
using MLDataUtils, CSV, LinearAlgebra
using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(joinpath(srcdir(),"intro.jl"))

## Parameters and data
nSamples = 100;
nChains = 5;
data = Matrix(CSV.read(joinpath(datadir(),"exp_raw/housing.csv"),header=false))
y = data[:,1]; X = data[:,2:end]; (N,nDim) = size(X)
l = initial_lengthscale(X)

kernel = RBFKernel(l)
K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+1e-4*I
burnin=1
ν = 10.0; likelihood = StudentTLikelihood(ν); noisegen = TDist(ν); lname="StudentT"
likelihood = LaplaceLikelihood(); noisegen = Laplace(); lname = "Laplace";
# likelihood = LogisticLikelihood(); noisegen = Normal(); lname = "Logistic"
if lname == "Laplace"
    global b = 1.0
    global C()=1/(2b)
    global g(y) = 0.0
    global α(y) = y^2
    global β(y) = 2*y
    global γ(y) = 1.0
    global φ(r) = exp(-sqrt(r)/b)
    global ∇φ(r) = -exp(-sqrt(r)/b)/(2*b*sqrt(r))
    global ll(y,x) = 1/(2b)*exp(-abs(y-x)/b)
elseif lname == "StudentT"
    global C()= gamma(0.5*(ν+1))/(sqrt(ν*π)*gamma(0.5*ν))
    global g(y) = 0.0
    global α(y) = y^2
    global β(y) = 2*y
    global γ(y) = 1.0
    global φ(r) = (1+r/ν)^(-0.5*(ν+1))
    global ∇φ(r) = -(0.5*(1+ν)/ν)*(1+r/ν)^(-0.5*(ν+1)-1)
    global ll(y,x) = pdf(LocationScale(y,1.0,TDist(ν)),x)
elseif lname == "Logistic"
    global C()= 0.5
    global g(y) = 0.5*y
    global α(y) = 0.0
    global β(y) = 0.0
    global γ(y) = 1.0
    global φ(r) = sech.(0.5*sqrt.(r))
    global ∇φ(r) = -0.25*(sech.(0.5*sqrt.(r))*tanh.(0.5*sqrt.(r)))/(sqrt.(r))
    global ll(y,x) = logistic(y*x)
end
AGP.@augmodel(St,Regression,C,g,α,β, γ, φ, ∇φ)

Statistics.var(::StLikelihood) = ν/(ν-2.0)

AGP.@augmodel(La,Regression,C,g,α,β, γ, φ, ∇φ)

Statistics.var(::LaLikelihood) = 2*b^2

var(StLikelihood())
function AGP.grad_quad(likelihood::LaLikelihood{T},y::Real,μ::Real,σ²::Real,inference::Inference) where {T<:Real}
    nodes = inference.nodes*sqrt2*sqrt(σ²) .+ μ
    Edlogpdf = dot(inference.weights,AGP.grad_log_pdf.(likelihood,y,nodes))
    Ed²logpdf =  zero(T)#1/(b * sqrt(twoπ*σ²))
    return -Edlogpdf::T, Ed²logpdf::T
end
AGP.@augmodel(Lo,Classification,C,g,α,β, γ, φ, ∇φ)

AGP.hessian_log_pdf(::LoLikelihood{T},y::Real,f::Real) where {T<:Real} = -exp(y*f)/AGP.logistic(-y*f)^2

AGP.grad_log_pdf(::LoLikelihood{T},y::Real,f::Real) where {T<:Real} = y*AGP.logistic(-y*f)

##
if lname == "Logistic"
    global genlikelihood = LoLikelihood()
elseif lname == "Laplace"
    global genlikelihood = LaLikelihood()
elseif lname == "StudentT"
    global genlikelihood = StLikelihood()
end


nIter = 100
mu = zeros(N,nIter)
sigma = zeros(N,N,nIter)
function cb(model,iter)
    mu[:,iter+1] = model.μ[1]
    sigma[:,:,iter+1] = model.Σ[1]
end


amodel = VGP(X,y,RBFKernel(0.1),genlikelihood,AnalyticVI(),verbose=3,optimizer=false)
train!(amodel,iterations=nIter,callback=cb)

varω = zeros(N)

function varomega(c)
    ForwardDiff.gradient(x->∇φ(x[1]),[c])[1]/φ(c) - (∇φ(c)/φ(c))^2
end

for (i,r) in enumerate(amodel.likelihood.c²[1])
    varω[i] = varomega(r)
end

diffmu = mu.-amodel.μ[1];
diffsigma = sigma.-amodel.Σ[1];
convergence_matrix_mu = 2*Diagonal(β(y))*Diagonal(varω)*Diagonal(amodel.μ[1])*amodel.Σ[1];
convergence_matrix_sigma = zeros(N,N,N);
for i in 1:N, j in 1:N, k in 1:N
    convergence_matrix_sigma[i,j,k] = 2*γ(y[i])*varω[i]*amodel.Σ[1][j,i]*(amodel.Σ[1][i,k]+amodel.μ[1][i]*amodel.μ[1][k])
end


convergence_matrix_sigma = 2*kron(amodel.Σ[1]*(Diagonal(γ(y).*varω)),amodel.Σ[1]+amodel.μ[1]*transpose(amodel.μ[1]))

eps_0_mu = diffmu[:,1]
eps_tmax_mu = zeros(nIter);
eps_tmean_mu = zeros(nIter);
eps_0_sig = diffsigma[:,:,1]
eps_tmax_sig = zeros(nIter);
eps_tmean_sig = zeros(nIter);
for i in 1:nIter
    eps_tmax_mu[i] = maximum(abs.(eps_0_mu))
    eps_tmax_sig[i] = maximum(abs.(eps_0_sig))
    eps_tmean_mu[i] = norm(eps_0_mu)
    eps_tmean_sig[i] = norm(eps_0_sig)
    global eps_0_mu = convergence_matrix_mu*eps_0_mu
    eps_0_sig_old = copy(eps_0_sig)
    for j in 1:N, k in 1:N
        eps_0_sig[j,k] = sum(convergence_matrix_mu[i,j,k]*eps_0_sig_old[j,k] for i in 1:N)
    end
end
lambda_mu = maximum(eigen(convergence_matrix).values);



using Plots
##
maxdiffmu = maximum.(abs,eachcol(diffmu))
maxdiffmu = maxdiffmu[maxdiffmu.!=0]
p1 = plot(1:length(maxdiffmu),maxdiffmu,lab="Max Error mu",yaxis=:log)
plot!(1:nIter,eps_tmax_mu,lab="Max Bound") |> display

##
meandiffmu = norm.(eachcol(diffmu))
meandiffmu = meandiffmu[meandiffmu.!=0]
p2 = plot(1:length(meandiffmu),meandiffmu,lab="Mean Error mu",yaxis=:log)
plot!(1:nIter,eps_tmean_mu,lab="Mean Bound")

plot(p1,p2)
# plot!(1:nIter,maximum(eachcol(diffmu)),lab="Error sigma")

##

maxdiffsig = maximum.(abs,[diffsig[:,:,i] for i in 1:nIter])
maxdiffsig = maxdiffsig[maxdiffsig.!=0]
p3 = plot(1:length(maxdiffmu),maxdiffmu,lab="Max Error sig",yaxis=:log)
plot!(1:nIter,eps_tmax_sig),lab="Max Bound") |> display

##
meandiffsig = norm.([diffsig[:,:,i] for i in 1:nIter])
meandiffsig = meandiffsig[meandiffsig.!=0]
p4 = plot(1:length(meandiffsig),meandiffsig,lab="Mean Error sig",yaxis=:log)
plot!(1:nIter,eps_tmean_sig,lab="Mean Bound")
plot(p3,p4)

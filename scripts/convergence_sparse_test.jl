using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Statistics, ForwardDiff
using MLDataUtils, CSV, LinearAlgebra, LaTeXStrings, SpecialFunctions
using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(joinpath(srcdir(),"intro.jl"))

## Parameters and data
nSamples = 100;
nChains = 5;
data = Matrix(CSV.read(joinpath(datadir(),"exp_raw/housing.csv"),header=false))
# data = Matrix(CSV.read(joinpath(datadir(),"exp_raw/heart.csv"),header=false))
pointsused = 1:50;
y = data[pointsused,1]; rescale!(y,obsdim=1)
X = data[pointsused,2:end]; (N,nDim) = size(X); rescale!(X,obsdim=1)
l = initial_lengthscale(X)

##
kernel = RBFKernel(l)
K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+1e-4*I
burnin=1
ν = 10.0; likelihood = StudentTLikelihood(ν); noisegen = TDist(ν); lname="StudentT"
# likelihood = LaplaceLikelihood(); noisegen = Laplace(); lname = "Laplace";
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


nIter = 50
M = 20
eta1 = zeros(M,nIter)
eta2 = zeros(M^2,nIter)
function cb(model,iter)
    eta1[:,iter+1] = model.η₁[1]
    eta2[:,iter+1] = vec(model.η₂[1])
end

##

amodel = SVGP(X,y,RBFKernel(0.1),genlikelihood,AnalyticVI(),M,verbose=3,optimizer=false)
train!(amodel,iterations=nIter,callback=cb)

##
varω = zeros(N)
Σ = amodel.Σ[1]; η₂ = amodel.η₂[1]
μ = amodel.μ[1]; η₁ = amodel.η₁[1]
κ = amodel.κ[1]

function varomega(c)
    ForwardDiff.gradient(x->∇φ(x[1]),[c])[1]/φ(c) - (∇φ(c)/φ(c))^2
end

for (i,r) in enumerate(amodel.likelihood.c²[1])
    varω[i] = varomega(r)
end

diffeta1 = eta1.-amodel.η₁[1];
diffeta2 = eta2.-vec(amodel.η₂[1]);
diffeta = vcat(diffeta1,diffeta1)
J¹ = -Diagonal(varω.*(2*(κ*μ).*γ.(y).-β.(y)))*κ*Σ
J² = -reshape(2*Diagonal(varω)*κ*Σ,N,M,1).*reshape(κ*Σ,N,1,M)-
    reshape(2*Diagonal(varω.*(2*κ*μ.*γ.(y).-β.(y)))*κ*Σ,N,M,1).*reshape(μ,1,1,M)
# Jtest = zeros(N,M,M)
# @progress for i in 1:N, j in 1:M, k in 1:M
    # Jtest[i,j,k] = -varω[i]*(sum(2*Σ[l,j]*μ[k]*(-β(y[i])*κ[i,l] + 2γ(y[i])*κ[i,l]*sum(κ[i,p]*μ[p] for p in 1:M)) for l in 1:M) +
    # 2*γ(y[i])*sum(Σ[m,j]*Σ[k,n]*κ[i,m]*κ[i,n] for m in 1:M, n in 1:M))
# end
# Jtest
# count(J².!=0)
# count(Jtest .≈ J²)
length(Jtest)
##
eps_0_eta1 = diffeta1[:,1]
eps_tmax_eta1 = zeros(nIter);
eps_tmean_eta1 = zeros(nIter);
eps_0_eta2 = reshape(diffeta2[:,1],M,M)
eps_tmax_eta2 = zeros(nIter);
eps_tmean_eta2 = zeros(nIter);
eps_tmean = zeros(nIter)
eps_tmax = zeros(nIter)

@progress for i in 1:nIter
    # i%10 == 0 ? @info("iteration $i") : nothing
    eps_tmax_eta1[i] = maximum(abs.(eps_0_eta1))
    eps_tmax_eta2[i] = maximum(abs.(vec(eps_0_eta2)))
    eps_tmax[i] = maximum(abs.(vcat(eps_0_eta1,vec(eps_0_eta2))))
    eps_tmean_eta1[i] = norm(eps_0_eta1)
    eps_tmean_eta2[i] = norm(vec(eps_0_eta2))
    eps_tmean[i] = norm(vcat(eps_0_eta1,vec(eps_0_eta2)))
    global part_eps_1 = J¹*eps_0_eta1
    global part_eps_2 = [sum(J²[i,:,:].*eps_0_eta2) for i in 1:N]
    global eps_0_eta1 = transpose(κ)*Diagonal(β.(y))*(part_eps_1+part_eps_2)
    global eps_0_eta2 = -transpose(κ)*Diagonal(γ.(y))*Diagonal(part_eps_1+part_eps_2)*κ
end
# totalmatrix = vcat(hcat(Diagonal(β.(y))*J¹,Diagonal(β.(y))*J²),
                    # hcat(-Diagonal(γ.(y))*J¹,-Diagonal(γ.(y))*J²))
# lambda_eta1_max = maximum(abs.(real.(eigen(J¹).values)))
# lambda_eta1_min = minimum(abs.(real.(eigen(J¹).values)))
# maximum(abs.(real.(eigen(totalmatrix).values)))
###

using Plots; pyplot()
##
default(lw=3.0)
maxdiffeta1 = maximum.(abs,eachcol(diffeta1))
maxdiffeta1 = maxdiffeta1[maxdiffeta1.!=0]
p1 = plot(1:length(maxdiffeta1),maxdiffeta1,lab="Max Error eta 1",yaxis=:log,xlabel="t",ylabel=L"\max(|\eta_1^t-\eta_1^*|)")
plot!(1:nIter,eps_tmax_eta1,lab="Max Expected Bound")
# plot!(1:nIter,x->lambda_eta1_max^x,lab="Upper bound")
# plot!(1:nIter,x->abs(lambda_eta1_min)^x,lab="Lower bound")
meandiffeta1 = norm.(eachcol(diffeta1))
meandiffeta1 = meandiffeta1[meandiffeta1.!=0]
p2 = plot(1:length(meandiffeta1),meandiffeta1,lab="Mean Error eta 1",yaxis=:log,xlabel="t",ylabel=L"||\eta_1^t-\eta_1^*||")
plot!(1:nIter,eps_tmean_eta1,lab="Mean Bound")

plot(p1,p2) |> display
# plot!(1:nIter,maximum(eachcol(diffmu)),lab="Error sigma")

##

maxdiffeta2 = maximum.(abs,eachcol(diffeta2))
maxdiffeta2 = maxdiffeta2[maxdiffeta2.!=0]
p3 = plot(1:length(maxdiffeta2),maxdiffeta2,lab="Max Error eta 2",yaxis=:log,xlabel="t",ylabel=L"\max(|\eta_2^t-\eta_2^*|)")
plot!(1:nIter,eps_tmax_eta2,lab="Max Bound")
meandiffeta2 = norm.(eachcol(diffeta2))
meandiffeta2 = meandiffeta2[meandiffeta2.!=0]
p4 = plot(1:length(meandiffeta2),meandiffeta2,lab="Mean Error eta 2",yaxis=:log,xlabel="t",ylabel=L"||\eta_2^t-\eta_2^*||")
plot!(1:nIter,eps_tmean_eta2,lab="Mean Bound")
plot(p3,p4) |> display


##

maxdiffeta = maximum.(abs,eachcol(diffeta))
maxdiffeta = maxdiffeta[maxdiffeta.!=0]
p5 = plot(1:length(maxdiffeta),maxdiffeta,lab="Max Error eta 1",yaxis=:log,xlabel="t",ylabel=L"\max(|\eta^t-\eta^*|)")
plot!(1:nIter,eps_tmax,lab="Max Expected Bound")
# plot!(1:nIter,x->lambda_eta1_max^x,lab="Upper bound")
# plot!(1:nIter,x->abs(lambda_eta1_min)^x,lab="Lower bound")
meandiffeta = norm.(eachcol(diffeta))
meandiffeta = meandiffeta[meandiffeta.!=0]
p6 = plot(1:length(meandiffeta),meandiffeta,lab="Mean Error eta",yaxis=:log,xlabel="t",ylabel=L"||\eta^t-\eta^*||")
plot!(1:nIter,eps_tmean,lab="Mean Bound")

plot(p5,p6) |> display
# plot!(1:nIter,maximum(eachcol(diffmu)),lab="Error sigma")

##
plot(p1,p2,p3,p4) |> display
savefig(joinpath(@__DIR__,"Convergence_sparse.png"))

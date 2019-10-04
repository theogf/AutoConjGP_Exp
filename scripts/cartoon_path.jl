using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(joinpath(srcdir(),"intro.jl"))
using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Statistics, ForwardDiff
using MLDataUtils, CSV, LinearAlgebra, LaTeXStrings, SpecialFunctions
using Plots

## Parameters and data
problem = :classification
# problem = :regression
if problem == :classification
    data = Matrix(CSV.read(joinpath(datadir(),"datasets/classification/small/heart.csv"),header=false))

elseif problem == :regression
    data = Matrix(CSV.read(joinpath(datadir(),"datasets/regression/small/housing.csv"),header=false))
end
pointsused = :;
y = data[pointsused,1];
problem == :classification ? nothing  : rescale!(y,obsdim=1)
X = data[pointsused,2:end]; (N,nDim) = size(X); rescale!(X,obsdim=1)
l = initial_lengthscale(X)
#
kernel = RBFKernel(l)
kernel = RBFKernel(21.0,variance=0.55)
K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+1e-4*I
burnin=1
if problem == :classfication
    likelihood = LogisticLikelihood(); noisegen = Normal(); lname = "Logistic"
elseif problem == :regression
    ν = 10.0; likelihood = StudentTLikelihood(ν); noisegen = TDist(ν); lname="StudentT"
end
# likelihood = LaplaceLikelihood(); noisegen = Laplace(); lname = "Laplace";

##
if lname == "Logistic"
    global genlikelihood = GenLogisticLikelihood()
elseif lname == "Laplace"
    global genlikelihood = GenLaplaceLikelihood()
elseif lname == "StudentT"
    global genlikelihood = GenStudentTLikelihood()
end

inits = [-10,120]
nIter = 100
μ = zeros(1,nIter+1); μ[1] = inits[1]
Σ = zeros(1,nIter+1); Σ[1] = inits[2]
ELBO_val = zeros(1,nIter)
# pred_f = zeros(N,nIter)
# sig_f = zeros(N,nIter)
function cb(model,iter)
    μ[:,iter+2] = model.μ[1]
    Σ[:,iter+2] = diag(model.Σ[1])
    ELBO_val[iter+1] = ELBO(model)
    # pred_f[:,iter+1],sig_f[:,iter+1] = predict_f(model,X,covf=true)
end

##
amodel = SVGP(X,y,kernel,genlikelihood,AnalyticVI(),1,verbose=3,optimizer=false)
amodel.Z[1] .= mean(X,dims=1)
amodel.μ[1] .= [inits[1]]
amodel.Σ[1] .= [inits[2]]
train!(amodel,iterations=nIter,callback=cb)
##

μopt = copy(amodel.μ[1])
Σopt = copy(amodel.Σ[1])
function restore_model(model,μopt,Σopt)
    amodel.μ[1] .= μopt
    amodel.Σ[1] .= Σopt
end

restore_model(amodel,μopt,Σopt)
function create_grid(model,dim,limsμ,limsΣ,nGrid)
    @assert dim <= model.nSample
    dim=1
    μ_orig = copy(model.μ[1][dim])
    Σ_orig = copy(model.Σ[1][dim,dim])
    rangeμ = range(limsμ...,length=nGrid)
    rangeΣ = 10.0.^range(limsΣ...,length=nGrid)
    ELBO_grid = zeros(nGrid,nGrid)
    try
        for i in 1:nGrid, j in 1:nGrid
            @show (i,j)
            model.μ[1][dim] = rangeμ[i]
            model.Σ[1][dim,dim] = rangeΣ[j]
            # model.Σ[1] = -0.5*Symmetric(inv(model.η₂[1]))
            AGP.local_updates!(model)

            ELBO_grid[i,j] = ELBO(model)
        end
    catch e
        model.μ[1][dim] = μ_orig
        model.Σ[1][dim,dim] = Σ_orig
        rethrow(e)
    end
    model.μ[1][dim] = μ_orig
    model.Σ[1][dim,dim] = Σ_orig
    return ELBO_grid,rangeμ,rangeΣ
end
pyplot()
limsμ = max(abs(inits[1]),abs(μopt[1]))*1.1
limsΣ = max(abs(log10(inits[2])),abs(log10(Σopt[1])))*1.1
grid,rangeμ,rangeΣ = create_grid(amodel,dim_pick,(-limsμ,limsμ),(-limsΣ,limsΣ),30)
contour(rangeμ,log10.(rangeΣ),log10.(-grid),colorbar=false)
plot!(μ[:],log10.(Σ[:]),lw=3.0,color=:blue,xlims=(-limsμ,limsμ),ylims=(-limsΣ,limsΣ),lab="") |>display
savefig("test.png")

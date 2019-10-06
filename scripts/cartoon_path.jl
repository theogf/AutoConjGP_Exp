using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(joinpath(srcdir(),"intro.jl"))
using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Statistics, ForwardDiff
using MLDataUtils, CSV, LinearAlgebra, LaTeXStrings, SpecialFunctions
using Plots
gpflow = pyimport("gpflow")
likelihood_GD = Dict(:Logistic=>py"BernoulliLogit()",:Matern32=>py"Matern32()",
    :Laplace=>py"Laplace()",:StudentT=>py"gpflow.likelihoods.StudentT(3.0)")
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

inits = [-10,1]
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

function cbflow(model,session,iter,X_test,y_test,valsGD)
    a = Vector{Float64}(undef,3)
    a[1]=model.q_mu.read_value(session)[1]
    a[2]=(model.q_sqrt.read_value(session)[1])^2
    a[3] = session.run(model.likelihood_tensor)
    push!(valsGD,a)
end

##
amodel = SVGP(X,y,kernel,genlikelihood,AnalyticVI(),1,verbose=3,optimizer=false)
amodel.Z[1] .= mean(X,dims=1)
amodel.μ[1] .= [inits[1]]
amodel.Σ[1] .= [inits[2]]
train!(amodel,iterations=nIter,callback=cb)
##
valsGD = []
GD_kernel = gpflow.kernels.RBF(nDim,lengthscales=l,ARD=true)
gdmodel = gpflow.models.SVGP(X, Float64.(reshape(y,(length(y),1))),kern=deepcopy(GD_kernel),likelihood=likelihood_GD[Symbol(lname)],num_latent=1,Z=mean(X,dims=1),q_mu=[[inits[1]]])
# ,q_sqrt=reshape([sqrt(inits[2])],:,1)
run_grads_with_adam(gdmodel,nIter*100,[],[],valsGD,ind_points_fixed=true,kernel_fixed=true,callback=cbflow,Stochastic=false)
# gdmodel.q_mu.set_value()[1] = inits[1]
# gdmodel.q_sqrt.value[1] = sqrt(inits[2])
valsGD=copy(hcat(valsGD...)')
##
valsNGD = []
NGD_kernel = gpflow.kernels.RBF(nDim,lengthscales=l,ARD=true)
ngdmodel = gpflow.models.SVGP(X, Float64.(reshape(y,(length(y),1))),kern=deepcopy(GD_kernel),likelihood=likelihood_GD[Symbol(lname)],num_latent=1,Z=mean(X,dims=1),q_mu=[[inits[1]]])
# ,q_sqrt=reshape([sqrt(inits[2])],:,1)
run_nat_grads_with_adam(ngdmodel,nIter*100,[],[],valsNGD,ind_points_fixed=true,kernel_fixed=true,callback=cbflow,Stochastic=false)
# gdmodel.q_mu.set_value()[1] = inits[1]
# gdmodel.q_sqrt.value[1] = sqrt(inits[2])
valsNGD=copy(hcat(valsNGD...)')

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
limsμ = max(abs(inits[1]),abs(μopt[1]))*1.1
limsΣ = max(abs(log10(inits[2])),abs(log10(Σopt[1])))*1.1
grid,rangeμ,rangeΣ = create_grid(amodel,dim_pick,(-12,5),(-5,2),50)
##
contourf(rangeμ,log10.(rangeΣ),log10.(-grid)',colorbar=false,levels=50)
plot!(μ[:],log10.(Σ[:]),lw=3.0,color=colors[1],xlims=extrema(rangeμ),ylims=log10.(extrema(rangeΣ)),lab="AACI",legend=:bottomleft) |>display
plot!(valsGD[:,1],log10.(valsGD[:,2]),lw=3.0,color=colors[2],lab="ADAM VI") |>display
plot!(valsNGD[:,1],log10.(valsNGD[:,2]),lw=3.0,color=colors[5],lab="NGD VI") |>display
savefig("test.png")

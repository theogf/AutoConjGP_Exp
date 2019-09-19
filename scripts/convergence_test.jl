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
pointsused = :;
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

##
if lname == "Logistic"
    global genlikelihood = GenLogisticLikelihood()
elseif lname == "Laplace"
    global genlikelihood = GenLaplaceLikelihood()
elseif lname == "StudentT"
    global genlikelihood = GenStudentTLikelihood()
end


nIter = 50
eta1 = zeros(N,nIter)
eta2 = zeros(N,nIter)
function cb(model,iter)
    eta1[:,iter+1] = model.η₁[1]
    eta2[:,iter+1] = diag(model.η₂[1])
end

##

amodel = VGP(X,y,RBFKernel(0.1),genlikelihood,AnalyticVI(),verbose=3,optimizer=false)
train!(amodel,iterations=nIter,callback=cb)


##
varω = zeros(N)

function varomega(c)
    ForwardDiff.gradient(x->∇φ(x[1]),[c])[1]/φ(c) - (∇φ(c)/φ(c))^2
end

for (i,r) in enumerate(amodel.likelihood.c²[1])
    varω[i] = varomega(r)
end

diffeta1 = eta1.-amodel.η₁[1];
diffeta2 = eta2.-diag(amodel.η₂[1]);
diffeta = vcat(diffeta1,diffeta1)
J¹ = -Diagonal(varω.*(2*amodel.μ[1].*γ.(y).-β.(y)))*amodel.Σ[1];
J² = (-2Diagonal(varω)*amodel.Σ[1]).*(Diagonal(γ.(y))*amodel.Σ[1]+(2*Diagonal(γ.(y))*amodel.μ[1].-β.(y))*transpose(amodel.μ[1]))

##
eps_0_eta1 = diffeta1[:,1]
eps_tmax_eta1 = zeros(nIter);
eps_tmean_eta1 = zeros(nIter);
eps_0_eta2 = diffeta2[:,1]
eps_tmax_eta2 = zeros(nIter);
eps_tmean_eta2 = zeros(nIter);
eps_tmean = zeros(nIter)
eps_tmax = zeros(nIter)

@progress for i in 1:nIter
    # i%10 == 0 ? @info("iteration $i") : nothing
    eps_tmax_eta1[i] = maximum(abs.(eps_0_eta1))
    eps_tmax_eta2[i] = maximum(abs.(eps_0_eta2))
    eps_tmax[i] = maximum(abs.(vcat(eps_0_eta1,eps_0_eta2)))
    eps_tmean_eta1[i] = norm(eps_0_eta1)
    eps_tmean_eta2[i] = norm(eps_0_eta2)
    eps_tmean[i] = norm(vcat(eps_0_eta1,eps_0_eta2))
    part_eps_1 = J¹*eps_0_eta1
    part_eps_2 = J²*eps_0_eta2
    global eps_0_eta1 = Diagonal(β.(y))*(part_eps_1+part_eps_2)
    global eps_0_eta2 = -Diagonal(γ.(y))*(part_eps_1+part_eps_2)
end
totalmatrix = vcat(hcat(Diagonal(β.(y))*J¹,Diagonal(β.(y))*J²),
                    hcat(-Diagonal(γ.(y))*J¹,-Diagonal(γ.(y))*J²))
lambda_eta1_max = maximum(abs.(real.(eigen(J¹).values)))
lambda_eta1_min = minimum(abs.(real.(eigen(J¹).values)))
maximum(abs.(real.(eigen(totalmatrix).values)))
###
if Sys.CPU_NAME == "skylake"
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
    savefig(joinpath(@__DIR__,"Convergence.png"))
end

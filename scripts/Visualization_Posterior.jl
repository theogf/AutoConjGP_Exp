using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using DrWatson
using Plots; pyplot()
using TensorBoardLogger, Logging
using Zygote
quickactivate("/home/theo/experiments/AutoConj")
include(joinpath(srcdir(),"intro.jl"))

using KernelDensity

# rm(joinpath(datadir(),"sims"),force=true,recursive=true)
### Create Data


nDim = 1
nPoints = 100
nGrid = 100
nIter = 100
kernel = RBFKernel(0.1)
ν = 10.0; likelihood = StudentTLikelihood(ν); noisegen = TDist(ν)
# likelihood = LaplaceLikelihood(); noisegen = Laplace(); lname = "Laplace";
likelihood = LogisticLikelihood(); noisegen = Normal(); lname = "Logistic"


X = rand(nPoints,nDim)
s = sortperm(X[:])
# x1_test = collect(range(-0.05,1.05,length=nGrid))
x1_test = collect(range(0,1,length=nGrid))
x2_test = collect(range(0,1,length=nGrid))

if nDim == 1
    X_test = x1_test
    y_true = 10*sample_gaussian_process(vcat(X,X_test),kernel,1e-10)
else
end
noise = rand(noisegen,length(y_true))
if isa(likelihood,ClassificationLikelihood)
    # global mu_logit_normal = dot(q_weights,AGP.logistic.(q_nodes))
    # global var_logit_normal = dot(q_weight,AGP.logistic.(q_nodes).^2)-mu_logit_normal^2
    global y = (rand.(Bernoulli.(AGP.logistic.(y_true+noise))).-0.5)*2.0
else
    global y = y_true + noise
end
y_test = y[nPoints+1:end]; y = y[1:nPoints]
# p_y = sum(logpdf.(LocationScale.(y_true[1:nPoints],1.0,noisegen),y))


scatter(X,y,lab="Data")
plot!(X_test,y_true[nPoints+1:end],lab="Truth")
##
amodel = VGP(X[s,:],y[s],kernel,likelihood,AnalyticVI(),optimizer=false,verbose=3)
train!(amodel)

##
gmodel = VGP(X[s,:],y[s],kernel,likelihood,GibbsSampling(nBurnin=100,samplefrequency=3),optimizer=false,verbose=3)
train!(gmodel,iterations=10003)
samples = gmodel.inference.sample_store[1]
Asamples = hcat(samples...)
minA,maxA = extrema(AGP.logistic.(Asamples))
nRange = 100
kderange = collect(range(minA,maxA,length=nRange))

KDEs = Vector{UnivariateKDE}(undef,gmodel.nFeatures)
pdfKDEs = zeros(gmodel.nFeatures,nRange)
for j in 1:gmodel.nFeatures
    KDEs[j] = kde(AGP.logistic.(Asamples[j,:]))
    pdfKDEs[j,:] = pdf(KDEs[j],kderange)
end
contourf(X[s,:][:],kderange,pdfKDEs',colorbar=false,color=:blues) |> display
# p = plot()
# for i in 1:length(n)
# end
    # plot!(X[s,:],samples[i],lab="",alpha=0.3)
# p
##
emodel = VGP(X[s,:],y[s],kernel,likelihood,QuadratureVI(nGaussHermite=100),optimizer=false,verbose=3)
train!(emodel,iterations=10000)


## Plotting of the predictions
nSig=  1; lw = 3.0; alph=0.5
if isa(likelihood,RegressionLikelihood)
    scatter(X,y,lab="Data")
    plot!(X_test,y_true[nPoints+1:end],lab="Truth",color=1)

    y_pred_a, var_pred_a = proba_y(amodel,X_test)
    plot!(X_test,y_pred_a.+nSig*sqrt.(var_pred_a),fill=y_pred_a.-nSig*sqrt.(var_pred_a),alpha=0.3,lw=0.0,lab="",color=2)
    plot!(X_test,y_pred_a,color=2,lab="Analytical")

    y_pred_e, var_pred_e = proba_y(emodel,X_test)
    plot!(X_test,y_pred_e.+nSig*sqrt.(var_pred_e),fill=y_pred_e.-nSig*sqrt.(var_pred_e),alpha=0.3,lw=0.0,lab="",color=3)
    plot!(X_test,y_pred_e,color=3,lab="Numerical")

    y_pred_atrue. var_pred_atrue = proba_y(atruemodel,X_test)
    plot!(X_test,y_pred_atrue.+nSig*sqrt.(var_pred_atrue),fill=y_pred_atrue.-nSig*sqrt.(var_pred_atrue),alpha=0.3,lw=0.0,lab="",color=4)
    plot!(X_test,y_pred_atrue,color=4,lab="True Analytical")

    y_pred_etrue, var_pred_etrue = proba_y(emodeltrue,X_test)
    plot!(X_test,y_pred_etrue.+nSig*sqrt.(var_pred_etrue),fill=y_pred_etrue.-nSig*sqrt.(var_pred_etrue),alpha=0.3,lw=0.0,lab="",color=3)
    plot!(X_test,y_pred_etrue,color=3,lab="True Numerical") |> display
elseif isa(likelihood,ClassificationLikelihood)
    # plot()
    contourf(X[s,:][:],kderange,pdfKDEs',colorbar=false,color=:grays_r)
    scatter!(X,(y./2.0).+0.5,lab="Data",xlims=(0,1))

    plot!(X_test,AGP.logistic.(y_true[nPoints+1:end]),lab="Truth (Median)",color=1,lw=lw)

    y_pred_a, var_pred_a = proba_y(amodel,X_test)
    plot!(X_test,y_pred_a.+nSig*sqrt.(var_pred_a),fillrange=y_pred_a.-nSig*sqrt.(var_pred_a),alpha=alph,lw=0.0,lab="",color=2)
    plot!(X_test,y_pred_a,color=2,lab="Analytical",lw=lw)

    # y_pred_e, var_pred_e = proba_y(emodel,X_test)
    # plot!(X_test,y_pred_e.+nSig*sqrt.(var_pred_e),fillrange=y_pred_e.-nSig*sqrt.(var_pred_e),alpha=alph,lw=0.0,lab="",color=3)
    # plot!(X_test,y_pred_e,color=3,lab="Numerical",lw=lw) |> display
end


KL_logdet(amodel.Σ[1],emodel.Σ[1])
KL_trace(amodel.Σ[1],emodel.Σ[1])
KL_dot(amodel.μ[1],emodel.μ[1],emodel.Σ[1])
GaussianKL(amodel.μ[1],emodel.μ[1],amodel.Σ[1],emodel.Σ[1])

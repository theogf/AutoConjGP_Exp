using DrWatson
quickactivate("/home/theo/experiments/AutoConj")
include(joinpath(srcdir(),"intro.jl"))
using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Plots; pyplot()
using TensorBoardLogger, Logging
using Zygote
using StatsFuns


using KernelDensity

# rm(joinpath(datadir(),"sims"),force=true,recursive=true)
### Create Data


nDim = 1
nPoints = 50
nGrid = 200
nIter = 100
kernel = RBFKernel(0.1)
ν = 10.0; likelihood = StudentTLikelihood(ν); noisegen = TDist(ν)
# likelihood = LaplaceLikelihood(); noisegen = Laplace(); lname = "Laplace";
likelihood = LogisticLikelihood(); noisegen = Normal(); lname = "Logistic"


X = rand(nPoints,nDim)
s = sortperm(X[:])
# x1_test = collect(range(-0.05,1.05,length=nGrid))
x1_test = collect(range(-0.2,1.2,length=nGrid))
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
train!(gmodel,iterations=301)
samples = gmodel.inference.sample_store[1]
Asamples = hcat(samples...)

scatter(X[s,:],(y[s].+1.0)/2.0,lab="",markerstrokewidth=0.0,framestyle=:none)
plot!(X[s,:],[logistic.(s) for s in samples],lab="",alpha=2/length(samples),color=:black,lw=3.0)

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
# emodel = VGP(X[s,:],y[s],kernel,likelihood,QuadratureVI(nGaussHermite=100),optimizer=false,verbose=3)
# train!(emodel,iterations=10000)

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

#
# KL_logdet(amodel.Σ[1],emodel.Σ[1])
# KL_trace(amodel.Σ[1],emodel.Σ[1])
# KL_dot(amodel.μ[1],emodel.μ[1],emodel.Σ[1])
# GaussianKL(amodel.μ[1],emodel.μ[1],amodel.Σ[1],emodel.Σ[1])
##
datapointsize = 5.0


y_pred_a, var_pred_a = proba_y(amodel,X_test)
f_pred_a, varf_pred_a = predict_f(amodel,X_test,covf=true)
logitnormalpdf(x,m,s) = 1/(s*sqrt2π*x*(1-x))*exp(-(logit(x)-m)^2/(2*s^2))
nGridx = 200
gridx = range(0,1,length=nGridx)
pdflogit = zeros(nGrid,nGridx)
for i in 1:nGridx, j in 1:nGrid
    pdflogit[j,i] = logitnormalpdf(gridx[i],f_pred_a[j],sqrt(varf_pred_a[j]))
end
contourf(X_test,collect(gridx),pdflogit',color=:amp,colorbar=:false,levels=100,framestyle=:none,background_color=RGBA(0.0,0.0,0.0,0.0))
scatter!(X,(y./2.0).+0.5,lab="",xlims=extrema(X_test),markerstrokewidth=0.0,framestyle=:none,markersize=datapointsize,color=1)|>display
savefig(plotsdir("figures","vi_inference.png"))
# plot!(X_test,min.(y_pred_a.+nSig*sqrt.(var_pred_a),1.0),fillrange=max.(y_pred_a.-nSig*sqrt.(var_pred_a),0.0),alpha=0.5,lw=0.0,lab="",color=2)
# plot!(X_test,min.(y_pred_a.+2*sqrt.(var_pred_a),1.0),fillrange=max.(y_pred_a.-2*sqrt.(var_pred_a),0.0),alpha=0.3,lw=0.0,lab="",color=2)
# plot!(X_test,min.(y_pred_a.+3*sqrt.(var_pred_a),1.0),fillrange=max.(y_pred_a.-3*sqrt.(var_pred_a),0.0),alpha=0.3,lw=0.0,lab="",color=2)
# plot!(X_test,y_pred_a,color=2,lab="",lw=lw) |>display
##

scatter(X[s,:],(y[s].+1.0)/2.0,lab="",markerstrokewidth=0.0,framestyle=:none,markersize=datapointsize)
N_test = size(X_test,1)
K = kernelmatrix(X[s,:],gmodel.kernel[1])
k_star = kernelmatrix(reshape(X_test,:,1),X[s,:],gmodel.kernel[1])
pred_mean = [k_star*gmodel.invKnn[1]].*gmodel.inference.sample_store[1]
k_starstar = kernelmatrix(reshape(X_test,:,1),gmodel.kernel[1]) + 1e-6I
K̃ = Symmetric(k_starstar - k_star*inv(K+1e-6I)*transpose(k_star))
eigvals(K̃)
isposdef(K̃)
pred_y_g = []
for m in pred_mean
    for i in 1:10
        push!(pred_y_g,logistic.(rand(MvNormal(m,K̃))))
    end
end
plot!(X_test,pred_y_g,lab="",alpha=2/length(samples),color=:red,lw=3.0) |> display
savefig(plotsdir("figures","gibbs_inference.png"))

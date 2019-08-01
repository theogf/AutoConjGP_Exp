using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using DrWatson
using Plots; pyplot()
using TensorBoardLogger, Logging
quickactivate("/home/theo/experiments/AutoConj")
include(joinpath(srcdir(),"intro.jl"))


### Create Data

nDim = 1
nPoints = 200
nGrid = 100
nIter = 100
nSig = 2
kernel = RBFKernel(0.1)
ν = 10.0; likelihood = StudentTLikelihood(ν); noisegen = TDist(ν)
likelihood = LaplaceLikelihood(); noisegen = Laplace()
# likelihood = LogisticLikelihood(); noisegen = Normal()


X = rand(nPoints,nDim)
s = sortperm(X[:])
x1_test = collect(range(-0.05,1.05,length=nGrid))
x2_test = collect(range(0,1,length=nGrid))

if nDim == 1
    X_test = x1_test
    y_true = sample_gaussian_process(vcat(X,X_test),kernel,1e-10)
else
end
y = y_true + rand(noisegen,length(y_true))
# y = sign.(y_true + rand(noisegen,length(y_true)))
y_test = y[nPoints+1:end]; y = y[1:nPoints]

function cb(model,iter)
    # display(plot(heatmap(model.η₂[1]),heatmap(model.Σ[1])))
    # display(plot(X[s],model.inference.∇μE[1][s],title="$iter"))
    @info "Var" elbo=ELBO(model) kv=getvariance(model.kernel[1]) kl=getlengthscales(model.kernel[1])[1]
    @info "Gradients" gradeta1=model.inference.∇η₁[1] gradeta2=diag(model.inference.∇η₂[1])
    if hasproperty(model.inference.optimizer[1],:η)
        @info "Rate" eta=model.inference.optimizer[1].η
    end
    if iter%50 == 0
        p = scatter(X,y,lab="Data")
        plot!(X_test,y_true[nPoints+1:end],lab="Truth")
        y_pred_e, var_pred_e = proba_y(model,X_test)
        plot!(X_test,y_pred_e.+nSig*sqrt.(var_pred_e),fill=y_pred_e.-nSig*sqrt.(var_pred_e),alpha=0.3,lw=0.0,lab="",color=3)
        plot!(X_test,y_pred_e,color=3,lab="Numerical")
        display(p)
    end
end

scatter(X,y,lab="Data")
plot!(X_test,y_true[nPoints+1:end],lab="Truth")
##
amodel = SVGP(X,y,kernel,likelihood,AnalyticSVI(20),100,optimizer=true,verbose=3)
alg =  TBLogger(joinpath(datadir(),"sims/analytical"),tb_overwrite)
with_logger(alg) do
    train!(amodel,iterations=1000,callback=cb)
end
# y_pred_a = proba_y(amodel,X_test)
y_pred_a, var_pred_a = proba_y(amodel,X_test)
plot!(X_test,y_pred_a.+nSig*sqrt.(var_pred_a),fill=y_pred_a.-nSig*sqrt.(var_pred_a),alpha=0.3,lw=0.0,lab="",color=1)
plot!(X_test,y_pred_a,color=1,lab="Analytical")
# plot!(X_test,(y_pred_a.-0.5)*2,color=1,lab="Analytical")
##
rm(joinpath(datadir(),"sims/numerical"),force=true,recursive=true)
elg =  TBLogger(joinpath(datadir(),"sims/numerical"),tb_overwrite)

emodel = VGP(X[s,:],y[s],kernel,likelihood,QuadratureVI(nGaussHermite=100),optimizer=true,verbose=3)
# emodel = VGP(X[s,:],y[s],kernel,likelihood,QuadratureVI(nGaussHermite=1000,optimizer=AGP.RMSprop(η=0.0001)),optimizer=true,verbose=3)
# AGP.computeMatrices!(emodel)
# emodel.Σ[1] = amodel.Σ[1]
# emodel.μ[1] = amodel.μ[1]
# emodel.η₁[1] = amodel.η₁[1]
# emodel.η₂[1] = emodel.η₂[1]
# emodel.Σ[1] = emodel.Knn[1]
# emodel.η₂[1] = -0.5*inv(emodel.Σ[1])
with_logger(elg) do
train!(emodel,iterations=1000,callback=cb)
end
# emodel.verbose = 0; @profiler train!(emodel,iterations=10)

##
scatter(X,y,lab="Data")
plot!(X_test,y_true[nPoints+1:end],lab="Truth",color=1)
y_pred_a, var_pred_a = proba_y(amodel,X_test)
plot!(X_test,y_pred_a.+nSig*sqrt.(var_pred_a),fill=y_pred_a.-nSig*sqrt.(var_pred_a),alpha=0.3,lw=0.0,lab="",color=2)
plot!(X_test,y_pred_a,color=2,lab="Analytical")
y_pred_e, var_pred_e = proba_y(emodel,X_test)
plot!(X_test,y_pred_e.+nSig*sqrt.(var_pred_e),fill=y_pred_e.-nSig*sqrt.(var_pred_e),alpha=0.3,lw=0.0,lab="",color=3)
plot!(X_test,y_pred_e,color=3,lab="Numerical")


###
# w=  2.0
# plot(x->log(pdf(Laplace(w),x)))
# plot(x->Zygote.gradient(y->log(pdf(Laplace(y),x)),w)[1])

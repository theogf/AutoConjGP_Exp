using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using DrWatson
using Plots; pyplot()
using LinearAlgebra
using TensorBoardLogger, Logging
using Zygote, SpecialFunctions, TaylorSeries, StatsFuns
using KernelDensity
quickactivate("/home/theo/experiments/AutoConj")
include(joinpath(srcdir(),"intro.jl"))

# rm(joinpath(datadir(),"sims"),force=true,recursive=true)
### Create Data


function taylor_estimate(amodel,order=11)
    m = amodel.μ[1]
    S = diag(amodel.Σ[1])
    θ = amodel.likelihood.θ[1]
    c² = amodel.likelihood.c²[1]
    y = amodel.inference.y[1]
    t = Taylor1(Float64,order)
    diff = 0.0
    if order < 2
        return diff
    end
    for o in 2:order
        subsum = 0.0
        for i in 1:amodel.nFeatures
            subsum += differentiate(o,log(φ(-t+c²[i])))*differentiate(o,logj(t,m[i],S[i],y[i]))
            # subsum += differentiate(o,log(φ(t+c²[i])))*differentiate(o,logj(t,m[i],S[i]))
        end
        # diff += (-1)^o*subsum/(factorial(o)*2^o)
        # @show exp(-o*log(2.0)- lfactorial(o) + log(subsum))
        diff += exp(log(subsum)-lfactorial(o))
        # @show subsum, diff
        # @info "order $o/$order"  subsum=subsum diff=diff
    end
    # @show diff
    # return 0.5*sum((∇²φ.(amodel.likelihood,c²)-∇φ.(c²).^2).*(4*S.*(m.^2)+3*S.^2))
    return diff
end
function logj(t,m,Σ,y)
    # 0.5-t*Σ+(m^2*t)/(1-2*t*Σ)
    -0.5+γ(y)*Σ*t+t*(γ(y)*m^2-β(y)*(m-0.5*β(y)*Σ*t))/(1-2*γ(y)*Σ*t)+α(y)*t
end

function GaussianKL(μ_1::AbstractVector{T},μ_2::AbstractVector{T},Σ_1::Symmetric{T,Matrix{T}},Σ_2::Symmetric{T,Matrix{T}}) where {T<:Real}
    0.5*(KL_logdet(Σ_1,Σ_2)+KL_trace(Σ_1,Σ_2)+KL_dot(μ_1,μ_2,Σ_2)-length(μ_1
    ))
end

function KL_logdet(Σ_1,Σ_2)
    -logdet(Σ_1)+logdet(Σ_2)
end


function KL_trace(Σ_1,Σ_2)
    opt_trace(inv(Σ_2),Σ_1)
end

function KL_dot(μ_1,μ_2,Σ_2)
    dot(μ_2-μ_1,inv(Σ_2)*(μ_2-μ_1))
end



function cb(model,iter)
    # display(plot(heatmap(model.η₂[1]),heatmap(model.Σ[1])))
    # display(plot(X[s],model.inference.∇μE[1][s],title="$iter"))
    if isa(model.inference,AGP.AnalyticVI)
        @info "ELBO" ELBO=ELBO(model) ELL=AGP.expecLogLikelihood(model) GaussianKL=AGP.GaussianKL(model) KLomega=AugmentedKL(model)
        global model2 =  VGP(X[s,:],y[s],kernel,genlikelihood,QuadratureVI(nGaussHermite=100),optimizer=false,verbose=0)
        model2.μ .= model.μ; model2.η₁ .= model.η₁
        model2.Σ .= model.Σ; model2.η₂ .= model.η₂
        model2.invKnn .= model.invKnn
        if iter >= 10
            @info "ELBO2" ELBO=ELBO(model2) ΔELBO=ELBO(model2)-ELBO(model) Taylor=taylor_estimate(model,3) log_step_increment=0
        end
        @info "Gradients" gradeta1=model.inference.∇η₁[1] gradeta2=diag(model.inference.∇η₂[1]) log_step_increment=0
    else
        @info "ELBO"  ELBO=ELBO(model) ELL=AGP.expecLogLikelihood(model) GaussianKL=AGP.GaussianKL(model)
        @info "Gradients" gradeta1=model.inference.λ[1] gradeta2=model.inference.ν[1] log_step_increment=0
    end
    @info "Var" elbo=ELBO(model) kv=getvariance(model.kernel[1]) kl=getlengthscales(model.kernel[1])[1] log_step_increment=0

    @info "Rate" eta=model.inference.optimizer[1].η log_step_increment=0
    if iter%1000 == 0
        # p = scatter(X,y,lab="Data")
        # plot!(X_test,y_true[nPoints+1:end],lab="Truth")
        # y_pred_e, var_pred_e = proba_y(model,X_test)
        # plot!(X_test,y_pred_e.+nSig*sqrt.(var_pred_e),fill=y_pred_e.-nSig*sqrt.(var_pred_e),alpha=0.3,lw=0.0,lab="",color=3)
        # plot!(X_test,y_pred_e,color=3,lab="Numerical")
        # display(p)
    end
end

nDim = 1
nPoints = 100
nGrid = 100
nIter = 100
kernel = RBFKernel(0.1)
# ν = 10.0; likelihood = StudentTLikelihood(ν); noisegen = TDist(ν); lname="StudentT"
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

X = rand(nPoints,nDim)
s = sortperm(X[:])
x1_test = collect(range(-0.05,1.05,length=nGrid))
x2_test = collect(range(0,1,length=nGrid))

if nDim == 1
    X_test = x1_test
    y_true = sample_gaussian_process(vcat(X,X_test),kernel,1e-10)
else
end
noise = rand(noisegen,length(y_true))
if isa(genlikelihood,ClassificationLikelihood)
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
amodel = VGP(X[s,:],y[s],kernel,genlikelihood,AnalyticVI(),optimizer=false,verbose=3)
alg= TBLogger(joinpath(datadir(),"sims/analytical"))
with_logger(alg) do
    train!(amodel,iterations=100,callback=cb)
end
##
atruemodel = VGP(X[s,:],y[s],kernel,likelihood,AnalyticVI(),optimizer=false,verbose=3)
train!(atruemodel)

##
elg = TBLogger(joinpath(datadir(),"sims/numerical"))

emodel = VGP(X[s,:],y[s],kernel,genlikelihood,QuadratureVI(nGaussHermite=100),optimizer=false,verbose=3)
with_logger(elg) do
    train!(emodel,iterations=1000,callback=cb)
end
# emodel.verbose = 0; @profiler train!(emodel,iterations=10)
##
etruemodel = VGP(X[s,:],y[s],kernel,likelihood,QuadratureVI(nGaussHermite=100),optimizer=false,verbose=3)
train!(etruemodel,iterations=1000)


##
gmodel = VGP(X[s,:],y[s],kernel,likelihood,GibbsSampling(nBurnin=100,samplefrequency=3),optimizer=false,verbose=3)
train!(gmodel,iterations=10003)
samples = gmodel.inference.sample_store[1]
Asamples = hcat(samples...)
minA,maxA = extrema(Asamples)
nRange = 100
kderange = collect(range(minA,maxA,length=nRange))

KDEs = Vector{UnivariateKDE}(undef,gmodel.nFeatures)
pdfKDEs = zeros(gmodel.nFeatures,nRange)
for j in 1:gmodel.nFeatures
    KDEs[j] = kde(Asamples[j,:])
    pdfKDEs[j,:] = pdf(KDEs[j],kderange)
end
contourf(X[s,:][:],kderange,pdfKDEs',colorbar=false,color=:blues) |> display
## Plotting of the predictions
nSig=  2
if isa(genlikelihood,RegressionLikelihood)
    scatter(X,y,lab="Data")
    plot!(X_test,y_true[nPoints+1:end],lab="Truth",color=1)

    y_pred_a, var_pred_a = proba_y(amodel,X_test)
    plot!(X_test,y_pred_a.+nSig*sqrt.(var_pred_a),fill=y_pred_a.-nSig*sqrt.(var_pred_a),alpha=0.3,lw=0.0,lab="",color=2)
    plot!(X_test,y_pred_a,color=2,lab="Analytical")

    y_pred_e, var_pred_e = proba_y(emodel,X_test)
    plot!(X_test,y_pred_e.+nSig*sqrt.(var_pred_e),fill=y_pred_e.-nSig*sqrt.(var_pred_e),alpha=0.3,lw=0.0,lab="",color=3)
    plot!(X_test,y_pred_e,color=3,lab="Numerical") |> display

    y_pred_atrue, var_pred_atrue = proba_y(atruemodel,X_test)
    plot!(X_test,y_pred_atrue.+nSig*sqrt.(var_pred_atrue),fill=y_pred_atrue.-nSig*sqrt.(var_pred_atrue),alpha=0.3,lw=0.0,lab="",color=4)
    plot!(X_test,y_pred_atrue,color=4,lab="True Analytical")

    y_pred_etrue, var_pred_etrue = proba_y(etruemodel,X_test)
    plot!(X_test,y_pred_etrue.+nSig*sqrt.(var_pred_etrue),fill=y_pred_etrue.-nSig*sqrt.(var_pred_etrue),alpha=0.3,lw=0.0,lab="",color=5)
    plot!(X_test,y_pred_etrue,color=5,lab="True Numerical") |> display
elseif isa(genlikelihood,ClassificationLikelihood)
    scatter(X,(y./2.0).+0.5,lab="Data")

    plot!(X_test,AGP.logistic.(y_true[nPoints+1:end]),lab="Truth (Median)",color=1)
    y_pred_a, var_pred_a = proba_y(amodel,X_test)
    plot!(X_test,y_pred_a.+nSig*sqrt.(var_pred_a),fillrange=y_pred_a.-nSig*sqrt.(var_pred_a),alpha=0.3,lw=0.0,lab="",color=2)
    plot!(X_test,y_pred_a,color=2,lab="Analytical")

    y_pred_e, var_pred_e = proba_y(emodel,X_test)
    plot!(X_test,y_pred_e.+nSig*sqrt.(var_pred_e),fillrange=y_pred_e.-nSig*sqrt.(var_pred_e),alpha=0.3,lw=0.0,lab="",color=3)
    plot!(X_test,y_pred_e,color=3,lab="Numerical") |> display

    # y_pred_atrue, var_pred_atrue = proba_y(atruemodel,X_test)
    # plot!(X_test,y_pred_atrue.+nSig*sqrt.(var_pred_atrue),fillrange=y_pred_atrue.-nSig*sqrt.(var_pred_atrue),alpha=0.3,lw=0.0,lab="",color=4)
    # plot!(X_test,y_pred_atrue,color=4,lab="True Analytical")

    # y_pred_etrue, var_pred_etrue = proba_y(etruemodel,X_test)
    # plot!(X_test,y_pred_etrue.+nSig*sqrt.(var_pred_etrue),fillrange=y_pred_etrue.-nSig*sqrt.(var_pred_etrue),alpha=0.3,lw=0.0,lab="",color=5)
    # plot!(X_test,y_pred_etrue,color=5,lab="True Numerical") |> display
end



###
# w=  2.0
# plot(x->log(pdf(Laplace(w),x)))
# plot(x->Zygote.gradient(y->log(pdf(Laplace(y),x)),w)[1])

# bar(["-log p(y)"," - Augmented ELBO", "- ELBO"],[-p_y,-ELBO(amodel),-ELBO(emodel)],lab="")
bar(["- Augmented ELBO", "- ELBO"],[-ELBO(amodel),-ELBO(emodel)],lab="")


KL_logdet(amodel.Σ[1],emodel.Σ[1])
KL_trace(amodel.Σ[1],emodel.Σ[1])
KL_dot(amodel.μ[1],emodel.μ[1],emodel.Σ[1])
GaussianKL(amodel.μ[1],emodel.μ[1],amodel.Σ[1],emodel.Σ[1])

##
pa = heatmap(amodel.Σ[1],yaxis=:flip,title="Augmented")
pe = heatmap(emodel.Σ[1],yaxis=:flip,title="VI")
pg = heatmap(gmodel.Σ[1],yaxis=:flip,title="Gibbs")
pk = heatmap(gmodel.Knn[1],yaxis=:flip,title="Prior")
plot(pa,pe,pg,pk) |> display

##

##
# mu_grid = collect(-2:0.1:2)
# sig_grid = 10 .^ collect(-3:0.1:1)
#
# La = zeros(length(mu_grid),length(sig_grid))
# La2 = zeros(length(mu_grid),length(sig_grid))
# Le = zeros(length(mu_grid),length(sig_grid))
#
#
# @progress for (i,mu) in enumerate(mu_grid)
#     emodel.μ[1][1] = mu
#     amodel.μ[1][1] = mu
#     for (j,sig) in enumerate(sig_grid)
#         emodel.Σ[1][1,1] = sig
#         amodel.Σ[1][1,1] = sig
#         Le[i,j] = ELBO(emodel)
#         La[i,j] = ELBO(amodel)
#         @info "Before" IGKL=AGP.InverseGammaKL(amodel) EL=AGP.expecLogLikelihood(amodel) tot=-AGP.InverseGammaKL(amodel)+AGP.expecLogLikelihood(amodel)
#         AGP.local_updates!(amodel)
#         @info "After"  IGKL=AGP.InverseGammaKL(amodel) EL=AGP.expecLogLikelihood(amodel) tot=-AGP.InverseGammaKL(amodel)+AGP.expecLogLikelihood(amodel)
#         La2[i,j] = ELBO(amodel)
#     end
# end
#
# pe = contourf(mu_grid,sig_grid,Le',xlabel="mu",ylabel="Sigma",title="VI",yaxis=:log)
# # pe = surface(mu_grid,sig_grid,Le',xlabel="mu",ylabel="Sigma",title="VI",yaxis=:log)
# pa = contourf(mu_grid,sig_grid,La',xlabel="mu",ylabel="Sigma",title="Augmented VI",yaxis=:log)
# pa2 = contourf(mu_grid,sig_grid,La2',xlabel="mu",ylabel="Sigma",title="Augmented VI postlocal",yaxis=:log)
#
# plot(pe,pa,pa2)
#
# pdiffea = contourf(mu_grid,sig_grid,(Le-La)',xlabel="mu",ylabel="Sigma",title="VI-AugVI",yaxis=:log)
# pdiffea2 = contourf(mu_grid,sig_grid,(Le-La2)',xlabel="mu",ylabel="Sigma",title="VI-AugVI2",yaxis=:log)
# pdiffaa2 = contourf(mu_grid,sig_grid,(La-La2)',xlabel="mu",ylabel="Sigma",title="AugVI-AugVI2",yaxis=:log)
# plot(pdiffea,pdiffea2)


using TaylorSeries

t = Taylor1(Float64,10)
differentiate(10,log(φ(t+1.0)))

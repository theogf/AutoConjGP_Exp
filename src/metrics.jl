
function RMSE(y_pred::AbstractVector,y_test::AbstractVector)
    sqrt(mean(abs2,y_pred-y_test))
end

function GaussianNLL(y_pred::AbstractVector,σ_pred::AbstractVector,y_test::AbstractVector)
    -mean(logpdf.(Normal.(y_pred,σ_pred),y_test))
end

function GaussianLogLikelihood(y_pred,y_test,noise)
    return -(-0.5*norm(y_pred-y_test)/noise^2-0.5*length(y_test)*log(2*π*noise^2))
end

function LogisticNLL(y_pred::AbstractVector,y_test::AbstractVector)
    -mean(vcat(log.(y_pred[y_test.==1]),log.(1.0.-y_pred[y_test.==-1])))
end

function JSGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -N*0.25
    tot += 0.125*sum(sig./(sig_f)+(sig_f)./(sig) + (1.0./(sig_f)+1.0./(sig)).*((mu-f).^2))
end


function KLGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -0.5*N
    tot += 0.5*sum(log.(sig)-log.(sig_f)+(sig_f+(mu-f).^2)./sig)
    return tot
end

function sigpartKLGP(mu,sig,f,sig_f)
    return 0.5*sum(log.(sig))
end

function sigpart2KLGP(mu,sig,f,sig_f)
    return 0.5*sum(sig_f./(sig))
end

function mupartKLGP(mu,sig,f,sig_f)
    return sum(0.5*(mu-f).^2 ./sig)
end

function JSGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -N*0.25
    tot += 0.125*sum(sig./(sig_f)+(sig_f)./(sig) + (1.0./(sig_f)+1.0./(sig)).*((mu-f).^2))
end

function distance_from_optimum(X,y,kernel,likelihood,Z)
    model = SVGP(X,y,kernel,likelihood,AnalyticVI(),Autotuning=true,Zoptimizer=Adam(α=0.01),size(Z[1],1),verbose=2)
    model.Z .= deepcopy.(Z)
    train!(model,iterations = 500)
    global Z_new = model.Z[1]
    return norm(Z[1]-Z_new)
end

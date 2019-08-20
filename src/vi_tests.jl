using Distributions
using LinearAlgebra
using Statistics
using StatsFuns

function PSIS(model,y,S::Int=1000,M::Int=floor(Int64,min(S/5,3*sqrt(S))))
    q = MvNormal(model.μ[1],model.Σ[1])
    p = MvNormal(AGP.array(model.μ₀[1],length(y)),model.Knn[1])
        #Sample from q(f)
    f_s = rand(q,S)
        # Compute rₛ = p(fₛ,y)/q(fₛ)
    r_s = exp.(broadcast(f->sum(AGP.logpdf.(model.likelihood,y,f))+logpdf(p,f)-logpdf(q,f),eachcol(f_s)))

    r_m = sort(r_s)[(end-M+1):end]
    # Fit Generalized Pareto to the M largest rₛ
    pareto = GeneralizedPareto(fit_Pareto(r_m)...)

    return pareto.ξ
end

# Method from https://www.tandfonline.com/doi/pdf/10.1198/tech.2009.08017?needAccess=true
function fit_Pareto(x)
    x = sort(x)
    n = length(x)
    x_star = x[floor(Int, n / 4.0 + 0.5)]

    m::Int = 30 + floor(sqrt(n))

    θ = [1 / x[n] + (1 - sqrt(m / (j - 0.5))) / (3 * x_star) for j in 1:m]

    lx(a, x) = let
        a = -a
        k = [mean(log1p.(b .* x)) for b in a]
        @. log(a / k) - k - 1
    end

    l_θ = n * lx(θ, x)
    w_θ = 1.0 ./ [sum(exp.(l_θ .- l_θ[j])) for j in 1:m]
    θ_hat = sum(θ .* w_θ)

    ξ = mean(log1p.(-θ_hat .* x))
    σ = -ξ ./ θ_hat
    ξ = ξ * n / (n + 10.0) + 10.0 * 0.5 / (n + 10.0)

    return σ, ξ
end


function VSBC(modeltype,X,kernel,likelihood,inference,M::Int=100,iterations::Int=100)
    K = kernelmatrix(X,kernel)+1e-5I
    N = size(X,1)
    prior = MvNormal(zeros(N),K)
    p = zeros(N,M)
    @progress for j in 1:M
        @info "Progress $j/$M"
        f = rand(prior)
        y = (rand.(Bernoulli.(AGP.logistic.(f))).-0.5)*2
        model = modeltype(X,y,kernel,likelihood,inference,optimizer=false)
        train!(model,iterations=iterations)
        for i in 1:N
            p[i,j] = 1.0-cdf(Normal(model.μ[1][i],sqrt(model.Σ[1][i,i])),f[i])
        end
    end
    return p
end

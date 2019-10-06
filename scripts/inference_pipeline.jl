using DrWatson
quickactivate(joinpath(@__DIR__,".."))
# include(joinpath(srcdir(),"intro.jl"))
using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Statistics, StatsFuns, Distributions
using MLDataUtils, CSV, LinearAlgebra, LaTeXStrings, SpecialFunctions
using Plots; pyplot()
using LaTeXStrings


nPoints = 1
nGrid = 200
kernel = RBFKernel(0.1)
X = rand(nPoints,1)
sort!(X,dims=1)
X_test = collect(range(-0.2,1.2,length=nGrid))
K = kernelmatrix(X,kernel)


default(guidefontsize=18,tickfontsize=15,legendfontsize=20,titlefontsize=22.0)
## p(y|f)
nx = 200
extremaf= 4
xrange = range(-extremaf,extremaf,length=nx)
plot(xrange,logistic,lab=L"p(y=1\mid f\;)",lw=5.0,xlabel=L"f")|>display
savefig(plotsdir("figures","p_y_f.png"))
## p(y|f,ω)
y=1
nw = 200
wrange = range(0,1,length=nw)
pdfw = zeros(nx,nw)
augll(x,w,y) = 0.5*exp(0.5*(y*x-x^2*w))
for i in 1:nx, j in 1:nw
    pdfw[i,j] = augll(xrange[i],wrange[j],y)
end
contourf(xrange,wrange,pdfw',colorbar=false,color=:blues,title=L"p(y\mid f, \omega\;)",xlabel=L"f",ylabel=L"\omega")
savefig(plotsdir("figures","p_y_f_omega.png"))

##
y = 1
function condf(w,y)
    s = inv(2*w+0.5*inv(K[1]))
    m = s*(0.5*y)
    d = Normal(m,sqrt(s))
    return pdf.(d,xrange)
end
omega = 0.5
(pcondf = plot(xrange,condf(omega,y),lab="",title=latexstring("p(f\\mid\\omega=$omega,y=1)"),lw=5.0,xlabel=L"f")) |> display
function condw(f)
    c = abs(f)
    pdf.([AGP.PolyaGammaDist()],wrange,1,f)
end
f = 0.5
(pcondw = plot(wrange,condw(f),lab="",title=latexstring("p(\\omega\\mid f =$f)"),lw=5.0,xlabel=L"\omega"))|> display
plot(pcondf,pcondw,layout=(2,1))|>display
savefig(plotsdir("figures","cond_p_f_omega.png"))

##
jointpdf =zeros(nx,nw)
for i in 1:nx, j in 1:nw
    jointpdf[i,j]= augll(xrange[i],wrange[j],y)*pdf(AGP.PolyaGammaDist(),wrange[j],1,0)*pdf(Normal(0,sqrt(K[1])),xrange[i])
end
contourf(xrange,wrange,jointpdf',colorbar=false,xlabel=L"f",ylabel=L"\omega")
iter = 50
function gibbs_samp()
    f = -2.0
    w = 0.5
    fs = [f]
    ws = [w]
    for i in 1:iter
        if i%2==0
            w = AGP.draw(AGP.PolyaGammaDist(),1,abs(f))
        else
            s = inv(2*w+0.5*inv(K[1]))
            m = s*(0.5*y)
            f = rand(Normal(m,sqrt(s)))
        end
        push!(ws,w)
        push!(fs,f)
    end
    return fs,ws
end
fs,ws=gibbs_samp()
plot!(fs,ws,color=:red,lab="",alpha=10/iter,lw=3.0,markersize=8.0,marker=:cross,markerstrokewidth=0.0,xlims=extrema(xrange),ylims=extrema(wrange))|>display
savefig(plotsdir("figures","gibbs_path.png"))

##
mu_init = -2.0
sigma_init = 50.0
model = VGP(X,[1],kernel,LogisticLikelihood(),AnalyticVI())
mus = [mu_init]
sigmas = [sigma_init]
model.μ[1][1] = mu_init
model.Σ[1][1] = sigma_init
function cb(model,iter)
    push!(mus,model.μ[1][1])
    push!(sigmas,model.Σ[1][1])
end
train!(model,iterations=10,callback=cb)
grid,rangemu,rangesigma = create_grid(model,1,(-3,3),(-2,2),100)
# maxgrid = maximum(grid)
contourf(rangemu,rangesigma,log.(abs.(grid))',yaxis=:log,title=L"\mathcal{L}(\mathbf{m},\mathbf{S})",xlabel=L"\mathbf{m}",ylabel=L"\mathbf{S}",levels=100,colorbar=false)
plot!(mus,sigmas,color=1,lw=5.0,marker=:cross,markersize=8.0,xlims=extrema(rangemu),ylims=extrema(rangesigma),lab="") |> display

function create_grid(model,dim,limsμ,limsΣ,nGrid)
    @assert dim <= model.nSample
    dim=1
    μ_orig = copy(model.μ[1][dim])
    Σ_orig = copy(model.Σ[1][dim,dim])
    rangeμ = range(limsμ...,length=nGrid)
    rangeΣ = 10.0.^range(limsΣ...,length=nGrid)
    ELBO_grid = zeros(nGrid,nGrid)
    try
        @progress for i in 1:nGrid, j in 1:nGrid
            # @show (i,j)
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
savefig(plotsdir("figures","vi_path.png"))

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


default(guidefontsize=16,tickfontsize=15)
## p(y|f)
nx = 200
extremaf= 4
xrange = range(-extremaf,extremaf,length=nx)
plot(xrange,logistic,lab="")
y=1
## p(y|f,ω)
nw = 200
wrange = range(0,1,length=nw)
pdfw = zeros(nx,nw)
augll(x,w,y) = 0.5*exp(0.5*(y*x-x^2*w))
for i in 1:nx, j in 1:nw
    pdfw[i,j] = augll(xrange[i],wrange[j],y)
end
contourf(xrange,wrange,pdfw',colorbar=false,color=:blues,xlabel=L"f",ylabel=L"\omega")
##
y = 1
function condf(w,y)
    s = inv(2*w+0.5*inv(K[1]))
    m = s*(0.5*y)
    d = Normal(m,sqrt(s))
    return pdf.(d,xrange)
end
(pcondf = plot(xrange,condf(2.0,y),lab="")) |> display
function condw(f)
    c = abs(f)
    pdf.([AGP.PolyaGammaDist()],wrange,1,f)
end
(pcondw = plot(wrange,condw(0.5)))|> display

##
jointpdf =zeros(nx,nw)
for i in 1:nx, j in 1:nw
    jointpdf[i,j]= augll(xrange[i],wrange[j],y)*pdf(AGP.PolyaGammaDist(),wrange[j],1,0)*pdf(Normal(0,sqrt(K[1])),xrange[i])
end
contourf(xrange,wrange,jointpdf',colorbar=false,color=:blues,xlabel=L"f",ylabel=L"\omega")
function gibbs_samp()
    iter = 10
    fs = []
    ws = []
    f = -2.0
    w = 0.0
    for i in 1:iter
        if i%2==0
            w = draw(AGP.PolyaGammaDist(),1,abs(f))
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

using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using Distributions
using LinearAlgebra
using Plots; pyplot()

xrange = range(-5,5,length=200)

βval = 1.0
dist = Laplace(0.0,βval)
# distω = InverseGamma(1,1/(2βval^2))
distω = Exponential(1/(2βval^2))
# rangeω = sort!(rand(distω,10))
rangeω = [0.05,0.2,0.5,1.0,2.0,5.0]
# rangeω = [0.02,0.03,0.04,0.05,0.1,0.2,0.5,1.0]
sort!(rangeω,rev=true)
# weights = pdf.(distω,rangeω) |> x->x./length(xrange)
weights = pdf.(distω,rangeω)# |> x->x./(sum(x))
weights = 1.5
pdfω = [pdf.(Normal(0.0,sqrt(ω)),xrange) for ω in rangeω]
plot(x->pdf(distω,x))
##
p_scale = plot(xrange,x->pdf(dist,x),lab="p(y|f)",color=:red,framestyle=:box,xtickfont=0,ytickfont=0,xticks=[],yticks=[],xlims=extrema(xrange),ylims=extrema(pdf.(dist,xrange)).+(0,0.01))
totpdf = zeros(length(xrange))
for (i,ω) in enumerate(rangeω)
    # global totpdf .+= pdfω[i]/weights[i]/length(rangeω)
    # plot!(xrange,totpdf,lab="",lw=1.0)
    plot!(xrange,x->pdf(Normal(0.0,sqrt(ω)),x)*weights/length(rangeω),lab="",lw=1.0,color=:black)
end
display(p_scale)
savefig(p_scale,plotsdir("figures","scalemixture.png"))

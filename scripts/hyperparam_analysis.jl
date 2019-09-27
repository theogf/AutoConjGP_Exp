using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles
using DataFramesMeta
using LaTeXStrings
using Plots; pyplot()

metric_name = Dict("Matern32"=>"RMSE","StudentT"=>"RMSE","Laplace"=>"RMSE","Logistic"=>"Accuracy")

likelihood = "Matern32"
file_name = likelihood == "Logistic" ? "heart" : "housing"
variance = 0.1
res = collect_results(datadir("part_2",file_name))
results = vcat(res.analysis_results...)
results = @linq results |> where(:LIKELIHOOD .== Symbol("Gen",likelihood,"Likelihood")) |> where(:VARIANCE .== variance) |> where(:ELBO_VI .!= NaN )
s = sortperm(results.LENGTHSCALE)
##
default(lw=6.0,legendfontsize=22,tickfontsize=18,guidefontsize=20)
p_ELBO = plot(log10.(results.LENGTHSCALE[s]),-results.ELBO_VI[s],xlabel=L"\log_{10}\theta",ylabel="Negative ELBO",lab="VI")
plot!(log10.(results.LENGTHSCALE[s]),-results.ELBO_A[s],lab="Augmented VI",linestyle=:dash)
p_METRIC = plot(log10.(results.LENGTHSCALE[s]),results.METRIC_VI[s],xlabel=L"\log_{10}\theta",ylabel=metric_name[likelihood],lab="VI")
plot!(log10.(results.LENGTHSCALE[s]),results.METRIC_A[s],lab="Augmented VI",linestyle=:dash)
p_NLL = plot(log10.(results.LENGTHSCALE[s]),-results.NLL_VI[s],xlabel=L"\log_{10}\theta",ylabel="Negative Test Log Likelihood",lab="VI")
plot!(log10.(results.LENGTHSCALE[s]),-results.NLL_A[s],lab="Augmented VI",linestyle=:dash)
plot(p_ELBO,p_METRIC,p_NLL) |> display
param_plots = @ntuple(likelihood,variance)
savefig(p_ELBO,plotsdir("part_2",savename("ELBO",param_plots,"png")))
savefig(p_METRIC,plotsdir("part_2",savename("METRIC",param_plots,"png")))
savefig(p_NLL,plotsdir("part_2",savename("NLL",param_plots,"png")))

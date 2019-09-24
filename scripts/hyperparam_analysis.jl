using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles
using DataFramesMeta
using LaTeXStrings
using Plots; pyplot()

metric_name = Dict("Matern32"=>"RMSE","StudentT"=>"RMSE","Laplace"=>"RMSE","Logistic"=>"Accuracy")

file_name = "heart"
likelihood = "Logistic"
variance = 0.01
res = collect_results(datadir("part_2",file_name))
results = vcat(res.analysis_results...)
results = @linq results |> where(:LIKELIHOOD .== Symbol("Gen",likelihood,"Likelihood")) |> where(:VARIANCE .== 0.01) |> where(:ELBO_VI .!= NaN )
s = sortperm(results.LENGTHSCALE)
default(lw=3.0,legendfontsize=15.0)
p_ELBO = plot(log10.(results.LENGTHSCALE[s]),-results.ELBO_A[s],xlabel=L"\theta",ylabel="Negative ELBO",lab="Augmented VI")
plot!(log10.(results.LENGTHSCALE[s]),-results.ELBO_VI[s],lab="VI")
p_METRIC = plot(log10.(results.LENGTHSCALE[s]),results.METRIC_A[s],xlabel=L"\theta",ylabel=metric_name[likelihood],lab="Augmented VI")
plot!(log10.(results.LENGTHSCALE[s]),results.METRIC_VI[s],lab="VI")
p_NLL = plot(log10.(results.LENGTHSCALE[s]),results.NLL_A[s],xlabel=L"\theta",ylabel="Negative Log Likelihood",lab="Augmented VI")
plot!(log10.(results.LENGTHSCALE[s]),results.NLL_VI[s],lab="VI")
plot(p_ELBO,p_METRIC,p_NLL)

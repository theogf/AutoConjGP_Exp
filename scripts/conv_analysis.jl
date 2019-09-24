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
likelihood = :Logistic
nInducing = 50; nMinibatch = 10
res = collect_results(datadir("part_3",file_name))
res.results
results = vcat(res.results...)
results = @linq results |> where(:likelihood .== likelihood)
default(lw=3.0,legendfontsize=15.0)
pelbo = plot(xlabel="t[s]",ylabel="Neg. ELBO")
pmetric =  plot(xlabel="t[s]",ylabel=metric_name[string(likelihood)])
pnll = plot(xlabel="t[s]",ylabel="Negative Log Likelihood")
for (i,type) in enumerate(results.model)
    res = @linq results |> where(:model.==type)
    t_init = (res.training_df[1].t_init.-res.t_first[1])./1e9
    t_end = (res.training_df[1].t_end.-res.t_first[1])./1e9
    t_eval = t_end .- cumsum(t_end-t_init)
    plot!(pelbo,t_eval,-res.training_df[1].ELBO,xaxis=:log,lab=type) |> display
    plot!(pmetric,t_eval,res.training_df[1].metric,xaxis=:log,lab=type) |> display
    plot!(pnll,t_eval,res.training_df[1].nll,xaxis=:log,lab=type) |> display
end

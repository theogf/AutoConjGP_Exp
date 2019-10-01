using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles
using DataFramesMeta
using LaTeXStrings
using Plots; pyplot()

metric_name = Dict("Matern32"=>"RMSE","StudentT"=>"RMSE","Laplace"=>"RMSE","Logistic"=>"Accuracy")

likelihood = :Matern32
file_name = likelihood == :Logistic ? "covtype" : "CASP"
x_axis = :time; nIter= 40000
# file_name = "covtype"
res = collect_results(datadir("part_3",file_name))
res.results
results = @linq vcat(res.results...) |> where(:likelihood .== likelihood)
default(lw=3.0,legendfontsize=15.0,dpi=600)
# for nInducing in [100,200,500], nMinibatch in [50,100,200]
    nInducing = 200; nMinibatch = 100;# nIter = 40000
    # @info "$((nInducing,nMinibatch))"
    refined_results = @linq results |> where(:nMinibatch .== nMinibatch) |> where(:nInducing .== nInducing)
    ##
    pelbo = plot(ylabel="Neg. ELBO")
    pmetric =  plot(ylabel=metric_name[string(likelihood)])
    pnll = plot(ylabel="Negative Log Likelihood")
    for (i,type) in enumerate(refined_results.model)
        res = @linq refined_results |> where(:model.==type)
        if x_axis == :iter
            iters = res.training_df[1].i
            plot!(pelbo,iters,-res.training_df[1].ELBO,xaxis=:log,lab=type,xlabel="Iterations")
            plot!(pmetric,iters,res.training_df[1].metric,xaxis=:log,lab=type,xlabel="Iterations")
            plot!(pnll,iters, -res.training_df[1].nll,xaxis=:log,lab=type,xlabel="Iterations")
        elseif x_axis == :time
            t_init = (res.training_df[1].t_init.-res.t_first[1])./1e9
            t_end = (res.training_df[1].t_end.-res.t_first[1])./1e9
            t_eval = t_end .- cumsum(t_end-t_init)
            plot!(pelbo,t_eval,-res.training_df[1].ELBO,xaxis=:log,lab=type,xlabel="t[s]")
            plot!(pmetric,t_eval,res.training_df[1].metric,xaxis=:log,lab=type,xlabel="t[s]")
            plot!(pnll,t_eval,-res.training_df[1].nll,xaxis=:log,lab=type,xlabel="t[s]")
        end
    end
    plot(pelbo,pmetric,pnll)|>display
    param_plots = @ntuple file_name likelihood x_axis nInducing nMinibatch
    savefig(pelbo,plotsdir("part_3",savename("ELBO",param_plots,"png")))
    savefig(pmetric,plotsdir("part_3",savename("METRIC",param_plots,"png")))
    savefig(pnll,plotsdir("part_3",savename("NLL",param_plots,"png")))
# end

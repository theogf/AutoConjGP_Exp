using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
include(joinpath(srcdir(),"plots_tools.jl"))
using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles
using DataFramesMeta
using LaTeXStrings
using Plots; pyplot()



metric_name = Dict("Matern32"=>"RMSE","StudentT"=>"RMSE","Laplace"=>"RMSE","Logistic"=>"Accuracy")
method_name = Dict(:CAVI=>"AACI (ours)",:NGD=>"NGD VI",:GD=>"ADAM SVI")


# for likelihood in [:Laplace,:Logistic,:Matern32,:StudentT]
    likelihood = :Logistic
    file_name = likelihood == :Logistic ? "covtype" : "CASP"
    x_axis = :time; nIter= 40000
    # file_name = "covtype"
    res = collect_results(datadir("part_3",file_name))
    results = @linq vcat(res.results...) |> where(:likelihood .== likelihood)
    default(lw=4.0,legendfontsize=18,dpi=600,tickfontsize=16,guidefontsize=18,legend=false)
    for nInducing in [100,200,500], nMinibatch in [50,100,200]
        # nInducing = 100; nMinibatch = 100;# nIter = 40000
        @info "$((nInducing,nMinibatch))"
        refined_results = @linq results |> where(:nMinibatch .== nMinibatch) |> where(:nInducing .== nInducing)
        if size(refined_results,1) == 0
            continue;
        end
        ##
        pelbo = plot(ylabel="Neg. ELBO")
        pmetric =  plot(ylabel=metric_name[string(likelihood)])
        pnll = plot(ylabel="Negative Log Likelihood")
        tlabel = "Training time  in seconds (log scale)"
        for (i,type) in enumerate(unique(refined_results.model))
            res = @linq refined_results |> where(:model.==type)
            if x_axis == :iter
                iters = res.training_df[1].i
                plot!(pelbo,iters,-res.training_df[1].ELBO,xaxis=:log,lab=method_name[type],xlabel="Iterations",color=colors[i])
                plot!(pmetric,iters,res.training_df[1].metric,xaxis=:log,lab=method_name[type],xlabel="Iterations",color=colors[i])
                plot!(pnll,iters, -res.training_df[1].nll,xaxis=:log,lab=method_name[type],xlabel="Iterations",color=colors[i])
            elseif x_axis == :time
                t_init = (res.training_df[1].t_init.-res.t_first[1])./1e9
                t_end = (res.training_df[1].t_end.-res.t_first[1])./1e9
                t_eval = t_end .- cumsum(t_end-t_init)
                plot!(pelbo,t_eval,-res.training_df[1].ELBO,xaxis=:log,lab=method_name[type],xlabel=tlabel,color=colors[i])
                plot!(pmetric,t_eval,res.training_df[1].metric,xaxis=:log,lab=method_name[type],xlabel=tlabel,color=colors[i])
                plot!(pnll,t_eval,-res.training_df[1].nll,xaxis=:log,lab=method_name[type],xlabel=tlabel,color=colors[i])
            end
        end
        plot(pelbo,pmetric,pnll)|>display
        param_plots = @ntuple file_name likelihood x_axis nInducing nMinibatch
        savefig(pelbo,plotsdir("part_3",savename("ELBO",param_plots,"png")))
        savefig(pmetric,plotsdir("part_3",savename("METRIC",param_plots,"png")))
        savefig(pnll,plotsdir("part_3",savename("NLL",param_plots,"png")))
    end
# end

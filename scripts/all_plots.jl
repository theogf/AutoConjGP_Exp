using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using Colors
include(joinpath(srcdir(),"plots_tools.jl"))
using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles
using DataFramesMeta
using LaTeXStrings
using Plots; pyplot()


nInducing = 200; nMinibatch = 100;  nIter= 50000
metric_name = Dict("Matern32"=>"RMSE","StudentT"=>"RMSE","Laplace"=>"RMSE","Logistic"=>"Accuracy")
method_name = Dict(:CAVI=>"AACI (ours)",:NGD=>"NGD VI",:GD=>"ADAM SVI")
like2dataset= Dict(:Logistic=>["covtype","SUSY"],:Laplace=>["airline","CASP"],:StudentT=>["airline","CASP"],:Matern32=>["airline","CASP"])
dataset2like = Dict("covtype"=>[:Logistic],"SUSY"=>[:Logistic],"airline"=>[:Laplace,:StudentT,:Matern32],"CASP"=>[:Laplace,:StudentT,:Matern32])

datasets_list = ["covtype","SUSY","CASP","CASP","airline"]
likelihood_list = [:Logistic,:Logistic,:Matern32,:Laplace,:StudentT]
@assert length(datasets_list)==length(likelihood_list)
default(lw=5.0,legendfontsize=18,titlefontsize=24,dpi=600,tickfontsize=16,guidefontsize=18,legend=false)
dataset_name = Dict("covtype"=>"Covtype","SUSY"=>"SUSY","CASP"=>"Protein","airline"=>"Airline","HIGGS"=>"HIGGS")

@info "Inducing,Minibatch : $((nInducing,nMinibatch))"
metricps=  []
nllps=  []
tlabel = "Training time  in seconds (log scale)"
for i in 1:length(datasets_list)
    @info "Doing dataset $(datasets_list[i]) with $(likelihood_list[i]) likelihood"
    res = collect_results(datadir("part_3",datasets_list[i]))
    x_axis = :time;
    likelihood = likelihood_list[i]
    # file_name = "covtype"
    global results = @linq vcat(res.results...) |> where(:likelihood .== likelihood) |> where(:nMinibatch .== nMinibatch) |> where(:nInducing .== nInducing)
    pmetric = plot(ylabel=i==1 ? "Test Error" : "")
    pnll = plot(ylabel=i==1 ? "Negative Test\n Log Likelihood" : "",title="$(dataset_name[datasets_list[i]]) ($(likelihood_list[i]))",legend=i==1 ? :topright : false)
    for (i,type) in enumerate(unique(results.model))
        res = @linq results |> where(:model.==type)
        # @show sample_i = findfirst(x->in(x.i[end],[nIter-10000,nIter]),res.training_df)
        if x_axis == :iter
            iters = res.training_df[1].i
            plot!(pmetric,iters,res.training_df[1].metric,xaxis=:log,lab=method_name[type],xlabel="",color=colors[i])
            plot!(pnll,iters, -res.training_df[1].nll,xaxis=:log,lab=method_name[type],xlabel="",color=colors[i])
        elseif x_axis == :time
            t_init = (res.training_df[1].t_init.-res.t_first[1])./1e9
            t_end = (res.training_df[1].t_end.-res.t_first[1])./1e9
            t_eval = t_end .- cumsum(t_end-t_init)
            plot!(pmetric,t_eval,res.training_df[1].metric,xaxis=:log,lab=method_name[type],xlabel="",color=colors[i])
            plot!(pnll,t_eval,-res.training_df[1].nll,xaxis=:log,lab=method_name[type],xlabel="",color=colors[i])
        end
    end
    push!(metricps,pmetric)
    push!(nllps,pnll)
    plot(pmetric,pnll)|>display
end
##
p_bottom = Plots.plot(annotation=(0.5,0.5,Plots.text("Training Time in Seconds (log scale)",font(22))),axis=:hide,grid=:hide)

l = @layout [Plots.grid(2,length(nllps),heights=[0.5,0.5])
                 p{0.05h}]
allps = vcat(nllps,metricps,[p_bottom])

(p = Plots.plot(allps...,layout=l,size=(1953,floor(Int64,0.7*850)),dpi=300)) |> display
param_plots = @ntuple nInducing nMinibatch
savefig(p,plotsdir("figures",savename("ConvPlot",param_plots,"png")))

using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles

res = collect_results(datadir("part_2","housing.csv"))
results = vcat(res.analysis_results...)
names(res.analysis_results[1])

results.ELBO_VI[results.ELBO_VI.==-Inf] .= 0


if Sys.CPU_NAME == "skylake"
    using Plots
    heatmap(log10.(list_l),log10.(list_v),reshape(results.ELBO_A,nGrid,nGrid),xlabel="log lengthscale",ylabel="log variance",title="ELBO Augmented")
    heatmap(log10.(list_l),log10.(list_v),reshape(results.ELBO_VI,nGrid,nGrid),xlabel="log lengthscale",ylabel="log variance",title="ELBO Classic")
    heatmap(log10.(list_l),log10.(list_v),reshape(results.METRIC_A,nGrid,nGrid),xlabel="log lengthscale",ylabel="log variance",title="Metric Augmented")
    heatmap(log10.(list_l),log10.(list_v),reshape(results.METRIC_VI,nGrid,nGrid),xlabel="log lengthscale",ylabel="log variance",title="Metric Classic")
    heatmap(log10.(list_l),log10.(list_v),reshape(results.NLL_A,nGrid,nGrid),xlabel="log lengthscale",ylabel="log variance",title="NLL Augmented")
    heatmap(log10.(list_l),log10.(list_v),reshape(results.NLL_VI,nGrid,nGrid),xlabel="log lengthscale",ylabel="log variance",title="NLL Classic")
    function merge_results(results::Vector{DataFrame})
        n = length(results)
        if n ==1
            return results[1]
        else
            colnames = names(results[1])
            m_results = DataFrame()
            for colname in colnames
                m = mean(results[i][colname] for i in 1:n)
                v= var.(eachrow(hcat([results[i][colname] for i in 1:n]...)))
                m_results = hcat(m_results,DataFrame([m,v],[Symbol(colname,"_μ"),Symbol(colname,"_σ")]))
            end
            return m_results
        end
    end
end

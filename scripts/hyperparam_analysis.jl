using DrWatson
quickactivate(joinpath("..", @__DIR__))
include(joinpath(srcdir(), "intro.jl"))
include(joinpath(srcdir(), "plots_tools.jl"))
using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles
using DataFramesMeta
using LaTeXStrings
using Plots;
pyplot();

metric_name = Dict(
    "Matern32" => "RMSE",
    "StudentT" => "RMSE",
    "Laplace" => "RMSE",
    "Logistic" => "Accuracy",
)

method_A = "VI (Augmented Model)"
method_VI = "VI (Original Model)"
for likelihood in ["Laplace", "Logistic", "Matern32", "StudentT"]
    # likelihood = "Laplace"
    file_name = likelihood == "Logistic" ? "heart" : "housing"
    variance = 0.1
    res = collect_results(datadir("part_2", file_name))
    results = vcat(res.analysis_results...)
    results = @linq results |>
          where(:LIKELIHOOD .== Symbol("Gen", likelihood, "Likelihood")) |>
          where(:VARIANCE .== variance) |>
          where(:ELBO_VI .!= NaN)
    s = sortperm(results.LENGTHSCALE)
    ##
    default(
        lw = 6.0,
        legendfontsize = 22,
        tickfontsize = 18,
        guidefontsize = 20,
        legend = :top,
        titlefontsize = 25,
    )
    p_ELBO = plot(
        log10.(results.LENGTHSCALE[s]),
        -results.ELBO_VI[s],
        xlabel = L"\log_{10}\theta",
        ylabel = "",
        title = "Negative ELBO",
        lab = method_VI,
        color = colors[1],
    )
    plot!(
        log10.(results.LENGTHSCALE[s]),
        -results.ELBO_A[s],
        lab = method_A,
        linestyle = :dash,
        ylims = (minimum(-results.ELBO_VI) * 0.95, maximum(-results.ELBO_A) * 1.15),
        color = colors[3],
    )
    p_METRIC = plot(
        log10.(results.LENGTHSCALE[s]),
        results.METRIC_VI[s],
        xlabel = L"\log_{10}\theta",
        ylabel = metric_name[likelihood],
        lab = method_VI,
        color = colors[1],
    )
    plot!(
        log10.(results.LENGTHSCALE[s]),
        results.METRIC_A[s],
        lab = method_A,
        linestyle = :dash,
        ylims = (minimum(results.METRIC_VI) * 0.95, maximum(results.METRIC_A) * 1.15),
        color = colors[3],
    )
    p_NLL = plot(
        log10.(results.LENGTHSCALE[s]),
        -results.NLL_VI[s],
        xlabel = L"\log_{10}\theta",
        ylabel = "",
        title = "Negative Test Log Likelihood",
        lab = method_VI,
        color = colors[1],
    )
    plot!(
        log10.(results.LENGTHSCALE[s]),
        -results.NLL_A[s],
        lab = method_A,
        linestyle = :dash,
        ylims = (minimum(-results.NLL_VI) * 0.95, maximum(-results.NLL_A) * 1.15),
        color = colors[3],
    )
    plot(p_ELBO, p_METRIC, p_NLL) |> display
    param_plots = @ntuple(likelihood, variance)
    savefig(p_ELBO, plotsdir("part_2", savename("ELBO", param_plots, "png")))
    savefig(p_METRIC, plotsdir("part_2", savename("METRIC", param_plots, "png")))
    savefig(p_NLL, plotsdir("part_2", savename("NLL", param_plots, "png")))
end

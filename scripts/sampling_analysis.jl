using DrWatson
quickactivate(joinpath("..", @__DIR__))
include(joinpath(srcdir(), "intro.jl"))
using Turing, MCMCChains
using DataFramesMeta
using MLDataUtils, CSV, StatsFuns
##
likelihood_name = "StudentT"
nSamples = 10000
epsilon = 0.05;
n_step = 10;

res = collect_results(datadir("part_1", likelihood_name), white_list = [:infos])
infos = vcat(res.infos...)
infosHMC = @linq infos |> where(:alg .== "HMC", :epsilon .== epsilon, :n_step .== n_step)
infosallHMC = @linq infos |> where(:alg .== "HMC")
infosMH = @linq infos |> where(:alg .== "MH", :nSamples .== nSamples)
infosGS = @linq infos |> where(:alg .== "GS", :nSamples .== nSamples)
dfresults = []
##
# paramsHMC = @ntuple nSamples
paramsHMC = @ntuple epsilon nSamples n_step
chain =
    DrWatson.wload(datadir("part_1", likelihood_name, savename("HMC", paramsHMC, "bson")))[:chains];
chain = unfold_chains(chain, nSamples);
@info "Chain loaded and unfolded"
dfchain = treat_chain(chain)
push!(dfresults, dfchain)
for alg in ["MH", "GS"]
    params = @ntuple nSamples
    chain =
        DrWatson.wload(datadir("part_1", likelihood_name, savename(alg, params, "bson")))[:chains]
    chain = unfold_chains(chain, nSamples)
    @info "Chain loaded and unfolded"
    dfchain = treat_chain(chain)
    push!(dfresults, dfchain)
end

# push!(dfresults,res)
savedf = deepcopy(dfresults)
results_infos = hcat(vcat(infosHMC, infosMH, infosGS), vcat(dfresults...))


using CSVFiles
save(plotsdir("part_1", likelihood_name, "results.csv"), results_infos)


function treat_chain(chain)
    burnin = mean(mean.(getindex.(heideldiag(chain), Symbol("Burn-in"))))
    gelman = mean(gelmandiag(chain)[:PSRF])
    autocorchain = autocor(chain, lags = collect(1:10))
    acorlag = zeros(10)
    for i in 1:10
        acorlag[i] = mean(mean.(abs, getindex.(autocorchain, Symbol("lag ", i))))
    end
    vals = vcat([[v] for v in acorlag], [[burnin], [gelman]])
    symbols = vcat([Symbol(:lag, i) for i in 1:10], [:mixtime, :gelman])
    return DataFrame(vals, symbols)
end

function unfold_chains(chain::DataFrame, nSamples::Int)
    nTot = length(chain[!, 1])
    @show nVar = length(names(chain))
    nChains = Int64(nTot / nSamples)
    samples = zeros(nSamples, nVar, nChains)
    for j in 1:nChains
        samples[:, :, j] = Matrix(chain[Int64.(((j-1)*nSamples+1):(j*nSamples)), :])
        # push!(fold_chains,Chains(reshape(Matrix(chain[Int64.(((j-1)*nSamples+1):(j*nSamples)),:]),nSamples,:,1)))
    end
    return Chains(samples)
end
## Summary HMC
dfresults = []
for epsilon in [0.01, 0.05, 0.1, 0.5], n_step in [1, 2, 5, 10]
    @show epsilon, n_step
    paramsHMC = @ntuple epsilon nSamples n_step
    chain = DrWatson.wload(datadir(
        "part_1",
        likelihood_name,
        savename("HMC", paramsHMC, "bson"),
    ))[:chains]
    chain = unfold_chains(chain, nSamples)
    @info "Chain loaded and unfolded"
    dfchain = treat_chain(chain)
    dfchain.epsilon = epsilon
    dfchain.n_step = n_step
    push!(dfresults, dfchain)
end
dfresults_old = deepcopy(dfresults)
results_infos = join(infosallHMC, vcat(dfresults...), on = [:epsilon, :n_step])
save(plotsdir("part_1", likelihood_name, "HMC_results.csv"), results_infos)

##
data = CSV.read(plotsdir("part_1", likelihood_name, "HMC_results.csv"))
data.table_string =
    "\\epsilon / n_{step}" * prod(" & $n" for n in sort(unique(data.n_step))) * "\\\\ \n "
for eps in sort(unique(data.epsilon))
    @show eps
    eps_data = @linq data |> where(:epsilon .== eps)
    display(size(eps_data))
    display(((@linq eps_data |> where(:n_step .== 10)).gelman))
    row_string = "$eps" * prod(" & " for n in sort(unique(eps_data.n_step))) * "\\\\ \n"
    global table_string = table_string * row_string
end
table_string

##
include(srcdir("plots_tools.jl"))
using Plots;
pyplot();
likelihood_name = "StudentT"
data = CSV.read(plotsdir("part_1", likelihood_name, "results.csv"))
epsilon = 0.05
n_steps = 10
lagindices = 8:(8+9)
data
lagGS = @linq data |> where(:alg .== "GS")
lagGS = Vector(lagGS[1, lagindices])
lagHMC = @linq data |>
      where(:alg .== "HMC") |>
      where(:epsilon .== epsilon) |>
      where(:n_step .== n_steps)
lagHMC = Vector(lagHMC[1, lagindices])
lagMH = @linq data |> where(:alg .== "MH")
lagMH = Vector(lagMH[1, lagindices])
default(
    lw = 4.0,
    legendfontsize = 18.0,
    guidefontsize = 15.0,
    tickfontsize = 15.0,
    titlefontsize = 18.0,
)
plot(
    xticks = collect(1:10),
    xlabel = "Lag",
    ylabel = "Correlation",
    legend = :right,
    title = likelihood_name * " Likelihood",
)
plot!(1:10, lagGS, lab = "Gibbs (ours)", color = colors[1])
plot!(1:10, lagHMC, lab = "HMC", linestyle = :solid, color = colors[2])
plot!(1:10, lagMH, lab = "MH", linestyle = :dash, color = colors[3]) |> display
savefig(plotsdir("part_1", likelihood_name, "lag_plot.png"))

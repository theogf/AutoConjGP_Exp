using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using Turing, MCMCChains
using AugmentedGaussianProcesses
using MLDataUtils, CSV, StatsFuns

likelihood_name = "Logistic"

res = collect_results(datadir("part_1",likelihood_name))
res[1]
function treat_chain(chain)
    burnin = mean(mean.(getindex.(heideldiag(chain),Symbol("Burn-in"))))
    gelman = mean(gelmandiag(chain)[:PSRF])
    acorlag1 = mean(mean.(abs,getindex.(autocor(chain),Symbol("lag 1"))))
    return burnin, gelman, acorlag1
end
treat_chain.(chains)
mean(mean.(getindex.(heideldiag(NUTSchain),Symbol("Burn-in"))))
## Diagnostics
mean(mean.(abs,getindex.(autocor(NUTSchain),Symbol("lag 1"))))
println("Gelman: NUTS : $(mean(gelmandiag(NUTSchain)[:PSRF])), GibbsSampling : $(mean(gelmandiag(GSchain)[:PSRF]))")
println("Heidel: NUTS : $(mean(mean.(getindex.(heideldiag(NUTSchain),Symbol("Burn-in"))))), GibbsSampling : $(mean(mean.(getindex.(heideldiag(GSchain),Symbol("Burn-in")))))")

function unfold_chains(results::DataFrame)
    infs = results.infos
    chains = results.chains
    unf_chains = []
    for i in 1:length(infs)
        nSamples = infs[i].nSamples[1]
        nTot = length(chains[i][!,1])
        nChains = nTot/nSamples
        global fold_chains = []
        for j in 1:nChains
            push!(fold_chains,Chains(reshape(Matrix(res.chains[i][Int64.(((j-1)*nSamples+1):(j*nSamples)),:]),nSamples,:,1)))
        end
        push!(unf_chains,chainscat(fold_chains...))
    end
    return unf_chains
end

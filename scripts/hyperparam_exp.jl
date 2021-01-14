using DrWatson
quickactivate(joinpath(@__DIR__, ".."))
include(joinpath(srcdir("intro.jl")))
using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles
using PyCall
using StatsFuns, SpecialFunctions
gpflow = pyimport("gpflow")
tf = pyimport("tensorflow")
## Create a default dictionary of parameters to go use
defaultdicthp = Dict(
    :nIterA => 30,
    :nIterVI => 100,
    :file_name => "housing",
    :kernel => RBFKernel,
    :likelihood => :Matern32,
    :AVI => true,
    :VI => true,
    :l => 1.0,
    :v => 10.0,
)

nGrid = 500 # Number of hyperparameters to train
list_l = 10.0 .^ (range(-2, 2, length = nGrid)) # Create a log scale grid for the lengthscale
list_v = [0.01, 0.1, 1.0, 10.0] # Create a simple log scale grid for the variance
alldicthp = Dict(
    :nIterA => 30,
    :nIterVI => 100,
    :file_name => "housing",
    :kernel => sqexponentialkernel,
    :likelihood => :Matern32,
    :AVI => true,
    :VI => true,
    :l => list_l,
    :v => list_v,
)
listdict_hp = dict_list(alldicthp)
problem_type = Dict(
    "covtype" => :classification,
    "heart" => :classification,
    "HIGGS" => :classification,
    "SUSY" => :classification,
    "CASP" => :regression,
    "housing" => :regression,
)
## Give mapping of the likelihood for the different techniques
like2like = Dict(
    :Matern32 => (GenMatern32Likelihood(), py"Matern32()"),
    :Laplace => (GenLaplaceLikelihood(), py"Laplace()"),
    :StudentT => (GenStudentTLikelihood(), py"gpflow.likelihoods.StudentT()"),
    :Logistic => (GenLogisticLikelihood(), py"BernoulliLogit()"),
)


##
function run_vi_exp_hp(dict::Dict = defaultdicthp)
    # Parameters and data
    nIterA = dict[:nIterA] # Number of iterations for the augmented model
    nIterVI = dict[:nIterVI] # Number of iterations for the vi model
    file_name = dict[:file_name] # Name of the dataset
    problem = problem_type[file_name]
    base_file = datadir("datasets", string(problem), "small", file_name)
    ## Load and preprocess the data
    data =
        isfile(base_file * ".h5") ? h5read(base_file * ".h5", "data") :
        Matrix(CSV.read(base_file * ".csv", header = false))
    y = data[:, 1]
    X = data[:, 2:end]
    (N, nDim) = size(X)
    rescale!(X, obsdim = 1)
    if length(unique(y)) != 2
        rescale!(y, obsdim = 1)
    end
    # l = initial_lengthscale(X)
    ktype = dict[:kernel]
    l = dict[:l]
    v = dict[:v]
    flowkernel = gpflow.kernels.RBF(nDim, lengthscales = l, variance = v, ARD = false)
    ker = v * ktype(l)
    likelihood = dict[:likelihood]
    ll, flowll = like2like[likelihood]
    kfold = 10 # Number of fold over the dataset
    nfold = kfold #Limit the number of k fold runs
    doAVI = dict[:AVI] # Flag for augmented model
    doVI = dict[:VI] # Flag for vi model
    predic_results = DataFrame()
    latent_results = DataFrame()
    i = 1
    elbo_a = 0.0
    elbo_vi = 0.0
    metric_a = 0.0
    metric_vi = 0.0
    nll_a = 0.0
    nll_vi = 0.0
    for ((X_train, y_train), (X_test, y_test)) in kfolds((X, y), obsdim = 1, k = kfold)
        # Computing truth
        @info "Using l=$l and v=$v, X_train:$(size(X))"
        predic_results = hcat(predic_results, DataFrame([y_test], [:y_test]))
        global save_y_train = copy(y_train) # save training outputs
        # Computing the augmented model
        if doAVI
            @info "Starting training of augmented model"
            global amodel =
                VGP(X_train, y_train, ker, ll, AnalyticVI(), verbose = 2, optimizer = false)
            train!(amodel, iterations = nIterA)
            y_a, sig_a = proba_y(amodel, X_test)
            predic_results = hcat(predic_results, DataFrame([y_a, sig_a], [:y_a, :sig_a]))
            latent_results = hcat(
                latent_results,
                DataFrame([amodel.μ[1], diag(amodel.Σ[1])], [:μ_a, :Σ_a]),
            )
            elbo_a = ELBO(amodel)
            metric_a = metric(ll, y_test, y_a, sig_a)
            nll_a = nll(ll, y_test, y_a, sig_a)
            println("ELBO is $elbo_a")
        end
        # Computing the classical model
        if doVI
            try
                @info "Starting training of classical model"
                vimodel = gpflow.models.VGP(
                    X_train,
                    y_train,
                    kern = flowkernel,
                    likelihood = flowll,
                    num_latent = 1,
                )
                run_nat_grads_with_adam(
                    vimodel,
                    nIterVI,
                    X_test,
                    y_test,
                    [],
                    Stochastic = false,
                )
                y_vi, sig_vi = proba_y(vimodel, X_test)
                sess = vimodel.enquire_session()
                predic_results =
                    hcat(predic_results, DataFrame([y_vi, sig_vi], [:y_vi, :sig_vi]))
                latent_results = hcat(
                    latent_results,
                    DataFrame(
                        [
                            vec(vimodel.q_mu.read_value(sess)),
                            diag(
                                reshape(
                                    vimodel.q_sqrt.read_value(sess),
                                    size(X_train, 1),
                                    size(X_train, 1),
                                ) |> x -> x * x',
                            ),
                        ],
                        [:μ_vi, :Σ_vi],
                    ),
                )
                elbo_vi = ELBO(vimodel, sess)
                metric_vi = metric(ll, y_test, y_vi, sig_vi)
                nll_vi = nll(ll, y_test, y_vi, sig_vi)
            catch e
                rethrow(e) #TODO treat error
                @show i += 1
                if i > nfold
                    break
                else
                    continue
                end
            end
        end
        i += 1
        if i > nfold
            break
        end
    end
    diff_elbo = elbo_vi - elbo_a
    analysis_results = DataFrame(
        [
            [l],
            [v],
            [nameof(typeof(ll))],
            [elbo_a],
            [elbo_vi],
            [diff_elbo],
            [metric_a],
            [metric_vi],
            [nll_a],
            [nll_vi],
        ],
        [
            :LENGTHSCALE,
            :VARIANCE,
            :LIKELIHOOD,
            :ELBO_A,
            :ELBO_VI,
            :DIFF_ELBO,
            :METRIC_A,
            :METRIC_VI,
            :NLL_A,
            :NLL_VI,
        ],
    )
    params = @ntuple(likelihood, l, v)
    @tagsave(
        datadir("part_2", file_name, savename(params, "bson")),
        @dict params predic_results latent_results analysis_results
    )
    return predic_results, latent_results, analysis_results
end

## Run this to experiment over a list of dictionaries
map(run_vi_exp_hp, listdict_hp)

## Run this to experiment over the default dictionary
_, lat, res = run_vi_exp_hp()

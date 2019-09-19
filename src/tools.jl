function initial_lengthscale(X)
    if size(X,1) > 10000
        D = pairwise(SqEuclidean(),X[sample(1:size(X,1),10000,replace=false),:],dims=1)
    else
        D = pairwise(SqEuclidean(),X,dims=1)
    end
    return sqrt(mean([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)]))
end

function run_nat_grads_with_adam(model,iterations; ind_points_fixed=true, kernel_fixed =true, callback=nothing , Stochastic = true)
    # we'll make use of this later when we use a XiTransform

    gamma_start = 1e-5;
    if Stochastic
        gamma_max = 1e-1;    gamma_step = 10^(0.1); gamma_fallback = 1e-3;
    else
        gamma_max = 1.0;    gamma_step = 10^(0.1); gamma_fallback = 1e-3;
    end
    gamma = tf.Variable(gamma_start,dtype=tf.float64);    gamma_incremented = tf.where(tf.less(gamma,gamma_max),tf.minimum(gamma*gamma_step,gamma_max),gamma_max)
    op_increment_gamma = tf.assign(gamma,gamma_incremented)
    op_gamma_fallback = tf.assign(gamma,gamma*gamma_fallback);
    sess = model.enquire_session();    sess.run(tf.variables_initializer([gamma]));
    var_list = [(model.q_mu, model.q_sqrt)]
    # we don't want adam optimizing these
    model.q_mu.set_trainable(false)
    model.q_sqrt.set_trainable(false)
    #
    # ind_points_fixed ? model.feature.set_trainable(false) : nothing
    kernel_fixed ? model.kern.set_trainable.(false) : nothing
    op_natgrad = gpflow.training.NatGradOptimizer.(gamma=gamma).make_optimize_tensor(model, var_list=var_list)
    op_adam=0

    if !(ind_points_fixed && kernel_fixed)
        op_adam = gpflow.train.AdamOptimizer().make_optimize_tensor(model)
    end

    for i in 1:(iterations)
        try
            sess.run(op_natgrad);sess.run(op_increment_gamma)
        catch e
          if isa(e,InterruptException)
                    println("Training interrupted by user at iteration $i");
                    break;
          else
            g = sess.run(gamma)
            # println("Gamma $g on iteration $i is too big: Falling back to $(g*gamma_fallback)")
            sess.run(op_gamma_fallback)
          end
        end
        if op_adam!=0
            sess.run(op_adam)
        end
        if i % 10 == 0
            println("$i gamma=$(sess.run(gamma)) ELBO=$(sess.run(model.likelihood_tensor))")
        end
        if callback!= nothing
            callback(model,sess,i)
        end
    end
    model.anchor(sess)
end

function AugmentedGaussianProcesses.proba_y(model,X_test)
    m_y,sig_y = model.predict_y(X_test)
    return vec(m_y),vec(sig_y)
end

function AugmentedGaussianProcesses.ELBO(model,sess=model.enquire_session())
    sess.run(model.likelihood_tensor)
end

function csv2h5(file_name::String,classification=false)
    @assert file_name[end-3:end] == ".csv" "Should be on a .csv file"
    file_short = file_name[1:end-4]
    data = readdlm(file_name,',')
    if classification
        y = data[:,1]
        ys = unique(y)
        length(ys) == 2
        y[y.==ys[1]] .= 1; y[y.==ys[2]] .= -1;
        data[:,1] .= y
    end
    if isfile(file_short*".h5")
        @warn "Rewriting file $(file_short*".h5")"
        rm(file_short*".h5")
    end
    h5write(file_short*".h5","data",data)
end

function convertall()
    top_folder = datadir("datasets")
    class_folds = readdir(joinpath(top_folder,"classification"))
    for fold in class_folds
        files = readdir(joinpath(top_folder,"classification",fold))
        for f in files
            csv2h5(joinpath(top_folder,"classification",fold,f),true)
        end
    end
    reg_folds = readdir(joinpath(top_folder,"regression"))
    for fold in class_folds
        files = readdir(joinpath(top_folder,"regression",fold))
        for f in files
            csv2h5(joinpath(top_folder,"regression",fold,f),false)
        end
    end
end

# using Plots
using JLD2
using Printf
using LinearAlgebra


function normalize_inputs!(inputs, norm_mode="none")
    if lowercase(norm_mode) == "standard"
        # normalize training data
        x_mu = mean(inputs, dims=2)
        x_std = std(inputs, dims=2)
        inputs[:] = (inputs .- x_mu) ./ (x_std .+ 1e-08)
        norm_factors = (x_mu, x_std) # tuple of Array{Float64,2}
    elseif lowercase(norm_mode) == "minmax"
        # normalize training data
        x_max = maximum(inputs, dims=2)
        x_min = minimum(inputs, dims=2)
        inputs[:] = (inputs .- x_min) ./ (x_max .- x_min .+ 1e-08)
        norm_factors = (x_min, x_max) # tuple of Array{Float64,2}
    else  # handles case of "", "none" or really any crazy string
        norm_factors = ([0.0], [1.0])
    end

    # to translate to unnormalized regression coefficients: m = mhat / stdx, b = bhat - (m*xmu)
    # precalculate a and b constants, and 
    # then just apply newvalue = a * value + b. a = (max'-min')/(max-min) and b = max - a * max 
    # (x - x.min()) / (x.max() - x.min())       # values from 0 to 1
    # 2*(x - x.min()) / (x.max() - x.min()) - 1 # values from -1 to 1

    return norm_factors
end


# apply previously used training normalization to a validation or test data set
function normalize_inputs!(inputs, norm_factors, norm_mode)
    if norm_mode == "standard"
        x_mu = norm_factors[1]
        x_std = norm_factors[2]
        inputs[:] = (inputs .- x_mu) ./ (x_std .+ 1e-08)
    elseif norm_mode == "minmax"
        x_min = norm_factors[1]
        x_max = norm_factors[2]
        inputs[:] = (inputs .- x_min) ./ (x_max .- x_min .+ 1e-08)
    else
        error("Input norm_mode = $norm_mode must be standard or minmax")
    end
end


####################################################################
#  functions to pre-allocate data updated during training loop
####################################################################

# use for test and training data
function preallocate_data!(dat, nnw, n, hp)

    # feedforward
    dat.a = [dat.inputs]  # allocates only tiny memory--it's a reference
    dat.z = [dat.inputs] # not used for input layer  TODO--this permeates the code but not needed
    if hp.sparse
        for i = 2:nnw.output_layer  
            push!(dat.z, spzeros(nnw.ks[i], n, 0.1))
            push!(dat.a, spzeros(nnw.ks[i], n, 0.1))  #  and up...  ...output layer set after loop
        end
    else
        for i = 2:nnw.output_layer  
            push!(dat.z, zeros(nnw.ks[i], n))
            push!(dat.a, zeros(nnw.ks[i], n))  #  and up...  ...output layer set after loop
        end
    end

    # training / backprop  -- pre-allocate only minibatch size (except last one, which could be smaller)
    # this doesn't work for test set when not using minibatches (minibatch size on training then > entire test set)
    # if istrain   # e.g., only for training->no backprop data structures needed for test data
        # if hp.dobatch   # TODO  fix this HACK
        #     dat.epsilon = [i[:,1:hp.mb_size_in] for i in dat.a]
        #     dat.grad = [i[:,1:hp.mb_size_in] for i in dat.a]
        #     dat.delta_z = [i[:,1:hp.mb_size_in] for i in dat.a]
        # else  # this should pick up sparsity
            # dat.epsilon = [i for i in dat.a]
            # dat.grad = [i for i in dat.a]
            # dat.delta_z = [i for i in dat.a]
        # end
    # end
    dat.epsilon = []
    dat.grad = []
    # dat.delta_z = []
    if hp.sparse
        for i = 1:nnw.output_layer  
            push!(dat.epsilon, spzeros(nnw.ks[i], n, 0.1))
            push!(dat.grad, spzeros(nnw.ks[i], n, 0.1))  #  and up...  ...output layer set after loop
            push!(dat.delta_z, spzeros(nnw.ks[i], n, 0.1))  #  and up...  ...output layer set after loop
        end
    else
        for i = 1:nnw.output_layer  
            push!(dat.epsilon, zeros(nnw.ks[i], n))
            push!(dat.grad, zeros(nnw.ks[i], n))  #  and up...  ...output layer set after loop
            # push!(dat.delta_z, zeros(nnw.ks[i], n))  #  and up...  ...output layer set after loop
        end
    end    

    if hp.do_batch_norm  # required for full pass performance stats  TODO: really? or only for batch_norm
        # feedforward
        dat.z_norm = deepcopy(dat.z)
        # backprop
        # dat.delta_z_norm = deepcopy(dat.z)
        # preallocate_bn_params!(bn, mb, nnw.ks)
    end

end


# method for batch views--currently the only method used
function preallocate_minibatch!(mb::Batch_view, nnw, hp)
    # feedforward:   minibatch views update the underlying data
    # TODO put @inbounds back after testing
    n_layers = nnw.output_layer

    # we don't need all of these depending on minibatches and batchnorm, but it's very little memory
    mb.a = Array{SubArray{}}(undef, n_layers)
    mb.targets = view([0.0],1:1)
    mb.z = Array{SubArray{}}(undef, n_layers)
    mb.z_norm = Array{SubArray{}}(undef, n_layers)
    # mb.delta_z_norm = Array{SubArray{}}(undef, n_layers)
    # mb.delta_z = Array{SubArray{}}(undef, n_layers)
    mb.grad = Array{SubArray{}}(undef, n_layers)
    mb.epsilon = Array{SubArray{}}(undef, n_layers)

end


####################################################################
#  functions to pre-allocate trained parameters
####################################################################


function preallocate_wgts!(nnw, hp, in_k, n, out_k)
    # initialize and pre-allocate data structures to hold neural net training data
    # theta = weight matrices for all calculated layers (e.g., not the input layer)
    # bias = bias term used for every layer but input
    # in_k = no. of features in input layer
    # n = number of examples in input layer (and throughout the network)
    # out_k = number of features in the targets--the output layer

    # theta dimensions for each layer of the neural network
    #    Follows the convention that rows = outputs of the current layer activation
    #    and columns are the inputs from the layer below

    # layers
    nnw.output_layer = 2 + size(hp.hidden, 1) # input layer is 1, output layer is highest value
    nnw.ks = [in_k, map(x -> x[2], hp.hidden)..., out_k]       # no. of output units by layer

    # set dimensions of the linear Wgts for each layer
    push!(nnw.theta_dims, (in_k, 1)) # weight dimensions for the input layer -- if using array, must splat as arg
    for l = 2:nnw.output_layer  
        push!(nnw.theta_dims, (nnw.ks[l], nnw.ks[l-1]))
    end

    # initialize the linear Wgts
    nnw.theta = [zeros(2,2)] # layer 1 not used

    # Xavier initialization--current best practice for relu
    if hp.initializer == "xavier"
        xavier_initialize!(nnw, hp.scale_init)
    elseif hp.initializer == "uniform"
        uniform_initialize!(nnw. hp.scale_init)
    elseif hp.initializer == "normal"
        normal_initialize!(nnw, hp.scale_init)
    else # using zeros generally produces poor results
        for l = 2:nnw.output_layer
            push!(nnw.theta, zeros(nnw.theta_dims[l])) # sqrt of no. of input units
        end
    end

    # bias initialization: small positive values can improve convergence
    nnw.bias = [zeros(2)] # this is layer 1: never used.  placeholder to make layer indices consistent

    if hp.bias_initializer == 0.0
        bias_zeros(nnw.ks, nnw)  
    elseif hp.bias_initializer == 1.0
        bias_ones(nnw.ks, nnw)
    elseif 0.0 < hp.bias_initializer < 1.0
        bias_val(hp.bias_initializer, nnw.ks, nnw)
    elseif np.bias_initializer == 99.9
        bias_rand(nnw.ks, nnw)
    else
        bias_zeros(nnw.ks, nnw)
    end

    # structure of gradient matches theta
    nnw.delta_th = deepcopy(nnw.theta)
    nnw.delta_b = deepcopy(nnw.bias)

    # initialize gradient, 2nd order gradient for Momentum or Adam or rmsprop
    if hp.opt == "momentum" || hp.opt == "adam" || hp.opt == "rmsprop"
        nnw.delta_v_th = [zeros(size(a)) for a in nnw.delta_th]
        nnw.delta_v_b = [zeros(size(a)) for a in nnw.delta_b]
    end
    if hp.opt == "adam"
        nnw.delta_s_th = [zeros(size(a)) for a in nnw.delta_th]
        nnw.delta_s_b = [zeros(size(a)) for a in nnw.delta_b]
    end

    # dropout
    if hp.dropout
        nnw.dropout_mask_units = [BitArray(ones(k)) for k in nnw.ks]
    end


end


function xavier_initialize!(nnw, scale=2.0)
    for l = 2:nnw.output_layer
        push!(nnw.theta, randn(nnw.theta_dims[l]...) .* sqrt(scale/nnw.theta_dims[l][2])) # sqrt of no. of input units
    end
end


function uniform_initialize!(nnw, scale=0.15)
    for l = 2:nnw.output_layer
        push!(nnw.theta, (rand(nnw.theta_dims[l]...) .- 0.5) .* (scale/.5)) # sqrt of no. of input units
    end        
end


function normal_initialize!(nnw, scale=0.15)
    for l = 2:nnw.output_layer
        push!(nnw.theta, randn(nnw.theta_dims[l]...) .* scale) # sqrt of no. of input units
    end
end


function bias_zeros(ks, nnw)
    for l = 2:nnw.output_layer
        push!(nnw.bias, zeros(ks[l]))
    end
end

function bias_ones(ks, nnw)
    for l = 2:nnw.output_layer
        push!(nnw.bias, ones(ks[l]))
    end
end

function bias_val(val, ks, nnw)
    for l = 2:nnw.output_layer
        push!(nnw.bias, fill(val, ks[l]))
    end
end

function bias_rand(ks, nnw)
    for l = 2:nnw.output_layer
        push!(nnw.bias, rand(ks[l]) .* 0.1)
    end
end


function preallocate_bn_params!(bn, mb, k)
    # initialize batch normalization parameters gamma and beta
    # vector at each layer corresponding to no. of inputs from preceding layer, roughly "features"
    # gamma = scaling factor for normalization standard deviation
    # beta = bias, or new mean instead of zero
    # should batch normalize for relu, can do for other unit functions
    # note: beta and gamma are reserved keywords, using bet and gam
    bn.gam = [ones(i) for i in k]  # gamma is a builtin function
    bn.bet = [zeros(i) for i in k] # beta is a builtin function
    bn.delta_gam = [zeros(i) for i in k]
    bn.delta_bet = [zeros(i) for i in k]
    bn.delta_v_gam = [zeros(i) for i in k]
    bn.delta_s_gam = [zeros(i) for i in k]
    bn.delta_v_bet = [zeros(i) for i in k]
    bn.delta_s_bet = [zeros(i) for i in k]
    bn.mu = [zeros(i) for i in k]  # same size as bias = no. of layer units
    bn.mu_run = [zeros(i) for i in k]
    bn.stddev = [zeros(i) for i in k]
    bn.std_run = [zeros(i) for i in k]
end


"""
Function setup_stats(hp, dotest::Bool)

Creates data structure to hold everything needed to plot progress of
neural net training by iteration.

Training statistics are tracked in a dict containing:

    "track"=>Dict of bools to select each type of results to be collected.
        Currently used are: "train", "test", "learning", "cost".  This determines what
        data will be collected during training iterations and what data series will be
        plotted.
    "labels"=>array of strings provides the labels to be used in the
        plot legend.
    "cost"=>array of calculated cost at each iteration
        with iterations as rows and data types ("train", "test") as columns.
    "accuracy"=>array of percentage of correct classification
        at each iteration with iterations as rows and result types as columns ("Training", "Test").
        This plots a so-called learning curve.  Very interesting indeed.
    "col_train"=>col_train: column of the arrays above to be used for Training results
    "col_test"=>col_test: column of the arrays above to be used for Test results
    "period"=>single string of "epoch" or "batch" chooses interval of data, or "" or "none" for none

"""
function setup_stats(hp, dotest::Bool)
    # set up cost_history to track 1 or 2 data series for plots
    # lots of indirection here:  someday might add "validation"
    if size(hp.stats,1) > 5
        @warn("Only 4 plot requests permitted. Proceeding with up to 4.")
    end

    valid_inputs = ["train", "test", "learning", "cost", "epoch", "batch"]
    if in(hp.stats, ["None", "none", ""])
        track = Dict(item => false for item in valid_stats) # set all to false
    else
        track = Dict(item => in(item, hp.stats) for item in valid_stats)
    end

    # determine whether to plot per batch or per epoch
    period = ""
    if in(hp.stats, ["None", "none", ""]) || isempty(hp.stats)
        period = ""      
    elseif in("epoch", hp.stats) # this is the default and overrides choosing both, which isn't supported
        period = "epoch"
    elseif in("batch", hp.stats) && hp.dobatch
        period = "batch"
    end

    pointcnt = if period == "epoch" 
                    hp.epochs 
                elseif period == "batch" 
                    hp.n_mb * hp.epochs
                else
                    0
                end

    # must have test data to plot test results
    if !dotest  # no test data
        if track["test"]  # input requested plotting test data results
            @warn("Can't plot test data. No test data. Proceeding.")
            track["test"] = false
        end
    end

    # set column in cost_history for each data series
    col_train = track["train"] ? 1 : 0
    col_test = track["test"] ? col_train + 1 : 0

    no_of_cols = max(col_train, col_test)

    labels = if col_train == 1 && col_test == 2
                ("Train", "Test")
            elseif col_train == 0 && col_test ==1
                ("Test",)    # trailing comma needed because a one-element tuple generates to its element
            elseif col_train == 1 && col_test == 0
                ("Train",)   # trailing comma needed because a one-element tuple generates to its element
            else
                ()
            end
    # labels = reshape(labels,1,size(labels,1)) # 1 x N row array required by pyplot

    # create all keys and values for dict stats
        stats = Dict("track"=>track, "labels"=>labels)

        if track["cost"]
            stats["cost"] = zeros(pointcnt, no_of_cols) # cost history initialized to 0's
        end
        if track["learning"]
            stats["accuracy"] = zeros(pointcnt, no_of_cols)
        end

        stats["col_train"] = col_train
        stats["col_test"] = col_test

        stats["period"] = period

    return stats
end
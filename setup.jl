using Plots
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


function setup_model!(mb, hp, nnw, bn, train)
    !hp.quiet && println("Setup_model beginning")
    !hp.quiet && println("hp.dobatch: ", hp.dobatch)

    # debug
    # println("norm_factors ", typeof(norm_factors))
    # println(norm_factors)

    #setup mini-batch
    if hp.dobatch
        @info("Be sure to shuffle training data when using minibatches.  Use utility function shuffle_data! or your own.")
        if hp.mb_size_in < 1
            hp.mb_size = hp.mb_size_in = train.n  
            hp.dobatch = false    # user provided incompatible inputs
        elseif hp.mb_size_in >= train.n
            hp.mb_size = hp.mb_size_in = train.n
            hp.dobatch = false   # user provided incompatible inputs
        else 
            hp.mb_size = hp.mb_size_in
        end
        hp.alphaovermb = hp.alpha / hp.mb_size  # calc once, use in hot loop
        hp.do_batch_norm = hp.dobatch ? hp.do_batch_norm : false  

    else
        hp.alphaovermb = hp.alpha / train.n 
    end


    hp.do_learn_decay = 
        if hp.learn_decay == [1.0, 1.0]
            false
        elseif hp.learn_decay == []
            false
        else
            true
        end

    hp.do_learn_decay && (hp.learn_decay = [learn_decay[1], floor(learn_decay[2])])

    # dropout parameters: droplim is in hp (Hyper_parameters),
    #    dropout_random and dropout_mask_units are in mb or train (Model_data)
    # set a droplim for each layer 
    if hp.dropout
        if length(hp.droplim) == length(hp.n_hid) + 2
            # droplim supplied for every layer
            if hp.droplim[end] != 1.0
                @warn("Poor performance when dropping units from output layer, continuing.")
            end
        elseif length(hp.droplim) == length(hp.n_hid) + 1
            # droplim supplied for input layer and hidden layers
            hp.droplim = [hp.droplim..., 1.0]  # keep all units in output layer
        elseif length(hp.droplim) < length(hp.n_hid)
            # pad to provide same droplim for all hidden layers
            for i = 1:length(hp.n_hid)-length(hp.droplim)
                push!(hp.droplim,hp.droplim[end]) 
            end
            hp.droplim = [1.0, hp.droplim..., 1.0] # use all units for input and output layers
        else
            @warn("More drop limits provided than total network layers, use limits ONLY for hidden layers.")
            hp.droplim = hp.droplim[1:length(hp.n_hid)]  # truncate
            hp.droplim = [1.0, hp.droplim..., 1.0] # placeholders for input and output layers
        end
    end

    # setup parameters for maxnorm regularization
    if titlecase(hp.reg) == "Maxnorm"
        hp.reg = "Maxnorm"
        if isempty(hp.maxnorm_lim)
            @warn("Values in Float64 array must be set for maxnormlim to use Maxnorm Reg, continuing without...")
            hp.reg = "L2"
        elseif length(hp.maxnorm_lim) == length(hp.n_hid) + 1
            hp.maxnorm_lim = append!([0.0], hp.maxnorm_lim) # add dummy for input layer
        elseif length(hp.maxnorm_lim) > length(hp.n_hid) + 1
            @warn("Too many values in maxnorm_lim; truncating to hidden and output layers.")
            hp.maxnorm_lim = append!([0.0], hp.maxnorm_lim[1:length(hp.n_hid)+1]) # truncate and add dummy for input layer
        elseif length(hp.maxnorm_lim) < length(hp.n_hid) + 1
            for i = 1:length(hp.n_hid)-length(hp.maxnorm_lim) + 1
                push!(hp.maxnorm_lim,hp.maxnorm_lim[end]) 
            end
            hp.maxnorm_lim = append!([0.0], hp.maxnorm_lim) # add dummy for input layer
        end
    end

    !hp.quiet && println("end of setup_model: hp.dobatch: ", hp.dobatch)

end


# iterator for minibatches of training examples
    struct MBrng
        cnt::Int
        incr::Int
    end

    function mbiter(mb::MBrng, state)
        up = state + mb.incr
        hi = up - 1 < mb.cnt ? up - 1 : mb.cnt
        ret = state < mb.cnt ? (state:hi, up) : nothing # return tuple of range and next state, or nothing--to stop iteration
        return ret
    end

    # add iterate method
    Base.iterate(mb::MBrng, state=1) = mbiter(mb::MBrng, state)

# verify the inputs in the toml file

function args_verify(argsdict)
    required = [:epochs, :n_hid]
    all(i -> i in Symbol.(keys(argsdict)), required) || error("Missing a required argument: epochs or n_hid")

    for (k,v) in argsdict
        checklist = get(valid_toml, Symbol(k), nothing)
        checklist === nothing && error("Parameter name is not valid: $k")

        for tst in checklist
            # eval the tst against the value in argsdict
            result = all(tst.f(v, tst.check)) 
            warn = get(tst, :warn, false)
            msg = get(tst, :msg, warn ? "Input argument not ideal: $k: $v" : "Input argument not valid: $k: $v" )
            !result && (warn ? @warn(msg) : error(msg))
        end
    end
end

eqtype(item, check) = typeof(item) == check
ininterval(item, check) = check[1] .<= item .<= check[2] 
oneof(item::String, check) = lowercase(item) in check 
oneof(item::Real, check) = item in check
lengthle(item, check) = length(item) <= check 
lengtheq(item, check) = length(item) == check
lrndecay(item, check) = ininterval(item[1],check[1]) && ininterval(item[2], check[2])

# for plots
valid_plots = ["learning", "cost", "train", "test", "batch", "epoch"]
function checkplots(item, _)
    if length(item) > 1
        ok1 = all(i -> i in valid_plots, lowercase.(item)) 
        ok2 = allunique(item) 
        ok3 = !("batch" in item || "epoch" in item)  # can't have both, ok to have neither
        ok = all([ok1, ok2, ok3])
    elseif length(item) == 1
        ok = item[1] in ["", "none"] 
    else
        ok = true
    end
    return ok
end

# key for each input param; value is list of checks as tuples 
#     check keys are f=function, check=values, warn can be true--default is false, msg="something"
#     example:  :alpha =>  [(f=eqtype, check=Float64), (f=ininterval, check=(.000001, 9.0), warn=true)]
const     valid_toml = Dict(
          :epochs => [(f=eqtype, check=Int), (f=ininterval, check=(1,9999))],
          :n_hid =>  [(f=eqtype, check=Array{Int, 1}), (f=lengthle, check=11), 
                       (f=ininterval, check=(1,8192))],  # empty is only way to say none                    
          :alpha =>  [(f=eqtype, check=Float64), (f=ininterval, check=(.000001, 9.0), warn=true)],
          :reg =>  [(f=oneof, check=["l2", "l1", "maxnorm", "", "none"])],
          :maxnorm_lim =>  [(f=eqtype, check=Array{Float64, 1})],
          :lambda =>  [(f=eqtype, check=Float64), (f=ininterval, check=(0.0, 5.0))],
          :learn_decay =>  [(f=eqtype, check=Array{Float64, 1}), (f=lengtheq, check=2), 
                            (f=lrndecay, check=((.1,.99), (1.0,20.0)))],
          :mb_size_in =>  [(f=eqtype, check=Int), (f=ininterval, check=(0,1000))],
          :norm_mode => [(f=oneof, check=["standard", "minmax", "", "none"])] ,
          :dobatch =>  [(f=eqtype, check=Bool)],
          :do_batch_norm =>  [(f=eqtype, check=Bool)],
          :opt =>  [(f=oneof, check=["momentum", "rmsprop", "adam", "", "none"])],
          :opt_params =>  [(f=eqtype, check=Array{Float64,1}), (f=ininterval, check=(0.5,1.0))],
          :units =>  [(f=oneof, check=["sigmoid", "l_relu", "relu", "tanh"]), 
                      (f=oneof, check=["l_relu", "relu"], warn=true, 
                       msg="Better results obtained with relu using input and/or batch normalization. Proceeding...")],
          :classify => [(f=oneof, check=["softmax", "sigmoid", "logistic", "regression"])] ,
          :dropout =>  [(f=eqtype, check=Bool)],
          :droplim =>  [(f=eqtype, check=Array{Float64, 1}), (f=ininterval, check=(0.2,1.0))],
          :plots =>  [(f=checkplots, check=nothing)],
          :plot_now =>  [(f=eqtype, check=Bool)],
          :quiet =>  [(f=eqtype, check=Bool)],
          :initializer => [(f=oneof, check=["xavier", "uniform", "normal", "zero"], warn=true, 
                            msg="Setting to default: xavier")] ,
          :scale_init =>  [(f=eqtype, check=Float64)],
          :bias_initializer => [(f=eqtype, check=Float64, warn=true, msg="Setting to default 0.0"),
                                (f=ininterval, check=(0.0,1.0), warn=true, msg="Setting to default 0.0")],
          :sparse =>  [(f=eqtype, check=Bool)]
        )



function build_hyper_parameters(argsdict)
    # assumes args have already been verified

    hp = Hyper_parameters()  # hyper_parameters constructor:  sets defaults

    for (k,v) in argsdict
        setproperty!(hp, Symbol(k), v)
    end

    return hp
end

####################################################################
#  functions to pre-allocate data updated during training loop
####################################################################

# use for test and training data
function preallocate_data!(dat, nnw, n, hp; istrain=true)

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
    if istrain   # e.g., only for training->no backprop data structures needed for test data
        if hp.dobatch   # TODO  fix this HACK
            dat.epsilon = [i[:,1:hp.mb_size_in] for i in dat.a]
            dat.grad = [i[:,1:hp.mb_size_in] for i in dat.a]
            dat.delta_z = [i[:,1:hp.mb_size_in] for i in dat.a]
        else  # this should pick up sparsity
            dat.epsilon = [i for i in dat.a]
            dat.grad = [i for i in dat.a]
            dat.delta_z = [i for i in dat.a]
        end
    end

    if hp.do_batch_norm  # required for full pass performance stats  TODO: really? or only for batch_norm
        # feedforward
        dat.z_norm = deepcopy(dat.z)
        # backprop
        dat.delta_z_norm = deepcopy(dat.z)
        # preallocate_batchnorm!(bn, mb, nnw.ks)
    end

    # backprop / training
    if hp.dropout
        dat.dropout_random = [i[:,1:hp.mb_size_in] for i in dat.a]
        dat.dropout_mask_units = [BitArray(ones(size(i,1),hp.mb_size_in)) for i in dat.a]
    end

end


function preallocate_nn_weights!(nnw, hp, in_k, n, out_k)
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
    nnw.output_layer = 2 + size(hp.n_hid, 1) # input layer is 1, output layer is highest value
    nnw.ks = [in_k, hp.n_hid..., out_k]       # no. of output units by layer

    # set dimensions of the linear weights for each layer
    push!(nnw.theta_dims, (in_k, 1)) # weight dimensions for the input layer -- if using array, must splat as arg
    for l = 2:nnw.output_layer  
        push!(nnw.theta_dims, (nnw.ks[l], nnw.ks[l-1]))
    end

    # initialize the linear weights
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
    nnw.bias = 
        if hp.bias_initializer == 0.0
            bias_zeros(nnw.ks)  
        elseif hp.bias_initializer == 1.0
            bias_ones(nnw.ks)
        elseif 0.0 < hp.bias_initializer < 1.0
            bias_val(hp.bias_initializer, nnw.ks)
        elseif np.bias_initializer == 99.9
            bias_rand(nnw.ks)
        else
            bias_zeros(nnw.ks)
        end

    # structure of gradient matches theta
    nnw.delta_w = deepcopy(nnw.theta)
    nnw.delta_b = deepcopy(nnw.bias)

    # initialize gradient, 2nd order gradient for Momentum or Adam
    if hp.opt == "momentum" || hp.opt == "adam"
        nnw.delta_v_w = [zeros(size(a)) for a in nnw.delta_w]
        nnw.delta_v_b = [zeros(size(a)) for a in nnw.delta_b]
    end
    if hp.opt == "adam"
        nnw.delta_s_w = [zeros(size(a)) for a in nnw.delta_w]
        nnw.delta_s_b = [zeros(size(a)) for a in nnw.delta_b]
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


function bias_zeros(ks)
    [zeros(i) for i in ks]
end

function bias_ones(ks)
    [ones(i) for i in ks]
end

function bias_val(val,ks)
    [fill(val, i) for i in ks]
end

function bias_rand(ks)
    [rand(i) .* .1 for i in ks]
end

function preallocate_minibatch!(mb::Batch_view, nnw, hp)
    # feedforward:   minibatch views update the underlying data
    # TODO put @inbounds back after testing
    n_layers = nnw.output_layer

    # we don't need all of these depending on minibatches and batchnorm, but it's very little memory
    mb.a = Array{SubArray{}}(undef, n_layers)
    mb.targets = view([0.0],1:1)
    mb.z = Array{SubArray{}}(undef, n_layers)
    mb.z_norm = Array{SubArray{}}(undef, n_layers)
    mb.delta_z_norm = Array{SubArray{}}(undef, n_layers)
    mb.delta_z = Array{SubArray{}}(undef, n_layers)
    mb.grad = Array{SubArray{}}(undef, n_layers)
    mb.epsilon = Array{SubArray{}}(undef, n_layers)
    mb.dropout_random = Array{SubArray{}}(undef, n_layers)
    mb.dropout_mask_units = Array{SubArray{}}(undef, n_layers)

end


# method that MIGHT work with slices?
function preallocate_minibatch!(mb::Batch_slice, nnw, hp)
    ncols = hp.mb_size_in

    mb.a = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]  
    mb.targets = zeros(nnw.ks[nnw.output_layer], ncols)
    mb.z = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.z_norm  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.delta_z_norm  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.delta_z  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    mb.grad  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    mb.epsilon  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.dropout_random  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.dropout_mask_units  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]

    if hp.do_batch_norm
        mb.z_norm  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
        mb.delta_z_norm = deepcopy(mb.epsilon)  # similar z
        mb.delta_z = deepcopy(mb.epsilon)       # similar z
    end

    #    #debug
    # println("size of pre-allocated mb.delta_z_norm $(size(mb.delta_z_norm))")
    # for i in 1:size(mb.delta_z_norm,1)
    #     println("$i size: $(size(mb.delta_z_norm[i]))")
    # end
    # error("that's all folks....")

    if hp.dropout
        mb.dropout_random = deepcopy(mb.epsilon)
        push!(mb.dropout_mask_units,fill(true,(2,2))) # for input layer, not used
        for item in mb.dropout_random[2:end]
            push!(mb.dropout_mask_units,fill(true,size(item)))
        end
    end
end


function preallocate_batchnorm!(bn, mb, k)
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
    bn.mu = [zeros(i) for i in k]  # same size as bias = no. of layer units
    bn.mu_run = [zeros(i) for i in k]
    bn.stddev = [zeros(i) for i in k]
    bn.std_run = [zeros(i) for i in k]

end


"""
define and choose functions to be used in neural net training
"""
function setup_functions!(hp, train)
    !hp.quiet && println("Setup functions beginning")
    # make these function variables module level 
        # the layer functions they point to are all module level (in file layer_functions.jl)
        # these are just substitute names or aliases
        # don't freak out about the word global--makes the aliases module level, too.
    global unit_function!
    global gradient_function!
    global classify_function!
    global optimization_function!
    global cost_function
    global reg_function!

    unit_function! =
        if hp.units == "sigmoid"
            sigmoid!
        elseif hp.units == "l_relu"
            l_relu!
        elseif hp.units == "relu"
            relu!
        elseif hp.units == "tanh"
            tanh_act!
        end

    gradient_function! =
        if unit_function! == sigmoid!
            sigmoid_gradient!
        elseif unit_function! == l_relu!
            l_relu_gradient!
        elseif unit_function! == relu!
            relu_gradient!
        elseif unit_function! == tanh_act!
            tanh_act_gradient!
        end

    classify_function! = 
        if train.out_k > 1  # more than one output (unit)
            if hp.classify == "sigmoid"
                sigmoid!
            elseif hp.classify == "softmax"
                softmax!
            else
                error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
            end
        else
            if hp.classify == "sigmoid" || hp.classify == "logistic"
                logistic!  # for one output label
            elseif hp.classify == "regression"
                regression!
            else
                error("Function to classify output must be \"sigmoid\", \"logistic\" or \"regression\".")
            end
        end

    reg_function! =
        if hp.reg == "L2"
            l2_reg!
        elseif hp.reg == "L1"
            l1_reg!
        elseif hp.reg == "Maxnorm"
            maxnorm_reg!
        else
            no_reg
        end

    optimization_function! = 
        if hp.opt == "momentum"
            momentum!
        elseif hp.opt == "adam"
            adam!
        elseif hp.opt == "rmsprop"
            rmsprop!
        else
            no_optimization
        end

    # set cost function
    cost_function = 
        if hp.classify=="regression" 
            mse_cost 
        else
            cross_entropy_cost
        end
    !hp.quiet && println("Setup functions completed.")
end


"""
Function setup_plots(hp, dotest::Bool)

Creates data structure to hold everything needed to plot progress of
neural net training by iteration.

A plotdef is a dict containing:

    "plot_switch"=>plot_switch: Dict of bools for each type of results to be plotted.
        Currently used are: "Training", "Test", "Learning", "Cost".  This determines what
        data will be collected during training iterations and what data series will be
        plotted.
    "plot_labels"=>plot_labels: array of strings provides the labels to be used in the
        plot legend.
    "cost_history"=>cost_history: an array of calculated cost at each iteration
        with iterations as rows and result types ("Training", "Test") as columns.
    "accuracy"=>accuracy: an array of percentage of correct classification
        at each iteration with iterations as rows and result types as columns ("Training", "Test").
        This plots a so-called learning curve.  Very interesting indeed.
    "col_train"=>col_train: column of the arrays above to be used for Training results
    "col_test"=>col_test: column of the arrays above to be used for Test results

"""
function setup_plots(hp, dotest::Bool)
    # set up cost_history to track 1 or 2 data series for plots
    # lots of indirection here:  someday might add "validation"
    if size(hp.plots,1) > 5
        @warn("Only 4 plot requests permitted. Proceeding with up to 4.")
    end

    valid_plots = ["train", "test", "learning", "cost", "epoch", "batch"]
    if in(hp.plots, ["None", "none", ""])
        plot_switch = Dict(pl => false for pl in valid_plots) # set all to false
    else
        plot_switch = Dict(pl => in(pl, hp.plots) for pl in valid_plots)
    end

    # determine whether to plot per batch or per epoch
    if in(hp.plots, ["None", "none", ""])
        hp.plotperbatch = false
        hp.plotperepoch = false        
    elseif !hp.dobatch # no batches--must plot by epoch
        hp.plotperbatch = false
        hp.plotperepoch = true
    elseif in("epoch", hp.plots) # this is the default and overrides conflicting choice
        hp.plotperbatch = false
        hp.plotperepoch = true
    elseif in("batch", hp.plots)
        hp.plotperbatch = true
        hp.plotperepoch = false
    else # even when NOT plotting we still gather stats in the plotdef for default epoch
        hp.plotperbatch = false
        hp.plotperepoch = true
    end

    pointcnt = hp.plotperepoch ? hp.epochs : hp.n_mb * hp.epochs

    # must have test data to plot test results
    if !dotest  # no test data
        if plot_switch["test"]  # input requested plotting test data results
            @warn("Can't plot test data. No test data. Proceeding.")
            plot_switch["test"] = false
        end
    end

    plot_labels = dotest ? ["Train", "Test"] : ["Train"]
    plot_labels = reshape(plot_labels,1,size(plot_labels,1)) # 1 x N row array required by pyplot

    plotdef = Dict("plot_switch"=>plot_switch, "plot_labels"=>plot_labels)

    if plot_switch["cost"]
        plotdef["cost_history"] = zeros(pointcnt, size(plot_labels,2)) # cost history initialized to 0's
    end
    if plot_switch["learning"]
        plotdef["accuracy"] = zeros(pointcnt, size(plot_labels,2))
    end

    # set column in cost_history for each data series
    col_train = plot_switch["train"] ? 1 : 0
    col_test = plot_switch["test"] ? col_train + 1 : 0

    plotdef["train"] = col_train
    plotdef["test"] = col_test

    return plotdef
end

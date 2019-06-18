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


function setup_model!(mb, hp, nnp, bn, train)
    !hp.quiet && println("Setup_model beginning")

    # debug
    # println("norm_factors ", typeof(norm_factors))
    # println(norm_factors)

    #setup mini-batch
    if hp.dobatch
        if hp.mb_size_in < 1
            hp.mb_size_in = train.n  # use 1 (mini-)batch with all of the examples
            hp.mb_size = train.n
        elseif hp.mb_size_in >= train.n
            hp.mb_size_in = train.n
            hp.mb_size = train.n
        else 
            hp.mb_size = hp.mb_size_in
        end
        hp.n_mb = ceil(Int, train.n / hp.mb_size)  # number of mini-batches
        wholebatches = floor(Int, train.n / hp.mb_size)
        hp.last_batch = hp.n_mb == wholebatches ? hp.mb_size : train.n - (wholebatches * hp.mb_size)
        hp.alphaovermb = hp.alpha / hp.mb_size  # calc once, use in hot loop
        hp.do_batch_norm = hp.n_mb == 1 ? false : hp.do_batch_norm  # no batch normalization for 1 batch

        # randomize order of all training samples:
            # labels in training data often in a block, which will make
            # mini-batch train badly because a batch will not contain mix of target labels
        # this slicing is SLOW AS HELL if data huge. But, repeated view/slicing raises a smaller cost.
        # do it this way: b = view(m, :, sel[colrng])
        if hp.mb_size < train.n  && hp.shuffle
            mb.sel = randperm(train.n)
            # train.inputs[:] = train.inputs[:, select_index]
            # train.targets[:] = train.targets[:, select_index]
        end
    else
        hp.alphaovermb = hp.alpha / train.n 
    end

    # set parameters for Momentum or Adam optimization
    if hp.opt == "momentum" || hp.opt == "adam"
        if !(hp.opt_params == [])  # use inputs for opt_params
            # set b1 for Momentum and Adam
            if hp.opt_params[1] > 1.0 || hp.opt_params[1] < 0.5
                @warn("First opt_params for momentum or adam should be between 0.5 and 0.999. Using default")
                # nothing to do:  hp.b1 = 0.9 and hp.b2 = 0.999 and hp.ltl_eps = 1e-8
            else
                hp.b1 = hp.opt_params[1] # use the passed parameter
            end
            # set b2 for Adam
            if length(hp.opt_params) > 1
                if hp.opt_params[2] > 1.0 || hp.opt_params[2] < 0.9
                    @warn("second opt_params for adam should be between 0.9 and 0.999. Using default")
                else
                    hp.b2 = hp.opt_params[2]
                end
            end
        end
    else
        hp.opt = ""
    end

    hp.do_learn_decay = 
        if hp.learn_decay == [1.0, 1.0]
            false
        elseif hp.learn_decay == []
            false
        else
            true
        end

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
        if isempty(hp.maxnormlim)
            @warn("Values in Float64 array must be set for maxnormlim to use Maxnorm Reg, continuing without...")
            hp.reg = "L2"
        elseif length(hp.maxnormlim) > length(hp.n_hid) + 1
            @warn("Too many values in maxnormlim; truncating to hidden and output layers.")
            hp.maxnormlim = [0.0, hp.maxnormlim[1:length(hp.n_hid)+1]] # truncate and add dummy for input layer
        else
            hp.maxnormlim = [0.0, hp.maxnormlim] # add dummy for input layer
        end
    end


    # debug
    # println("opt params: $(hp.b1), $(hp.b2)")

    # DEBUG see if hyper_parameters set correctly
    # for sym in fieldnames(hp)
    #    println(sym," ",getfield(hp,sym), " ", typeof(getfield(hp,sym)))
    # end

end


function preallocate_storage!(hp, nnp, bn, mb, datalist)
    ##########################################################################
    #  pre-allocate data storage
    ##########################################################################
    !hp.quiet && println("Pre-allocate storage starting")

    if size(datalist, 1) == 1
        train = datalist[1]
        dotest = false
    elseif size(datalist, 1) == 2
        train = datalist[1]
        test = datalist[2]
        dotest = true
    else
        error("Size of datalist must be 1 or 2.")
    end

    preallocate_nn_params!(nnp, hp, train.in_k, train.n, train.out_k)
    preallocate_data!(train, nnp, train.n, hp)
    # batch normalization parameters
    hp.do_batch_norm && preallocate_batchnorm!(bn, mb, nnp.layer_units)

    # feedfwd test data--if test input found
    if dotest
        test.n = size(test.inputs,2)   # TODO move this to be close to where train.n is originally set
        istrain = false
        preallocate_data!(test, nnp, test.n, hp, istrain)
    end

    !hp.quiet && println("Pre-allocate storage completed")

    # debug
    # verify correct dimensions of dropout filter
    # for item in mb.dropout_mask_units
    #     println(size(item))
    # end
    # error("that's all folks!....")
    !hp.quiet && println("Setup model completed")
end



####################################################################
#  functions to pre-allocate data updated during training loop
####################################################################

# use for test and training data
function preallocate_data!(dat, nnp, n, hp, istrain=true)
    # feedforward

    dat.a = [dat.inputs]  # allocates only tiny memory--it's a reference
    dat.z = [dat.inputs] # not used for input layer  TODO--this permeates the code but not needed
    if hp.sparse
        for i = 2:nnp.output_layer-1  # hidden layers
            push!(dat.z, spzeros(nnp.layer_units[i], n, 0.1))
            push!(dat.a, spzeros(size(dat.z[i]), 0.1))  #  and up...  ...output layer set after loop
        end
        push!(dat.z, spzeros(size(nnp.theta[nnp.output_layer],1), n))
        push!(dat.a, zeros(size(nnp.theta[nnp.output_layer],1),n))
    else
        for i = 2:nnp.output_layer-1  # hidden layers
            push!(dat.z, zeros(nnp.layer_units[i], n))
            push!(dat.a, zeros(size(dat.z[i])))  #  and up...  ...output layer set after loop
        end
        push!(dat.z, zeros(size(nnp.theta[nnp.output_layer],1),n))
        push!(dat.a, zeros(size(nnp.theta[nnp.output_layer],1),n))
    end

    # training / backprop  -- pre-allocate only minibatch size (except last one, which could be smaller)
    # this doesn't work for test set when not using minibatches (minibatch size on training then > entire test set)
    if istrain   # e.g., only for training
        if hp.dobatch   # TODO  fix this HACK
            dat.epsilon = [i[:,1:hp.mb_size_in] for i in dat.a]
            dat.grad = [i[:,1:hp.mb_size_in] for i in dat.a]
            dat.delta_z = [i[:,1:hp.mb_size_in] for i in dat.a]
        else
            dat.epsilon = [i for i in dat.a]
            dat.grad = [i for i in dat.a]
            dat.delta_z = [i for i in dat.a]

        end
    end

    if hp.dobatch  # required for full pass performance stats  TODO: really? or only for batch_norm
        # feedforward
        dat.z_norm = deepcopy(dat.z)
        # backprop
        dat.delta_z_norm = [i[:,1:hp.mb_size_in] for i in dat.a]
        # preallocate_batchnorm!(bn, mb, nnp.layer_units)
    end

    # backprop / training
    if hp.dropout
        dat.dropout_random = [i[:,1:hp.mb_size_in] for i in dat.a]
        dat.dropout_mask_units = [BitArray(ones(size(i,1),hp.mb_size_in)) for i in dat.a]
    end

end


function preallocate_nn_params!(nnp, hp, in_k, n, out_k)
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
    nnp.output_layer = 2 + size(hp.n_hid, 1) # input layer is 1, output layer is highest value
    nnp.layer_units = [in_k, hp.n_hid..., out_k]

    # set dimensions of the linear weights for each layer
    push!(nnp.theta_dims, (in_k, 1)) # weight dimensions for the input layer -- if using array, must splat as arg
    for l = 2:nnp.output_layer-1  # l refers to nn layer so this includes only hidden layers
        push!(nnp.theta_dims, (nnp.layer_units[l], nnp.layer_units[l-1]))
    end
    push!(nnp.theta_dims, (out_k, nnp.layer_units[nnp.output_layer - 1]))  # weight dims for output layer: rows = output classes

    # initialize the linear weights
    nnp.theta = [zeros(2,2)] # layer 1 not used

    # Xavier initialization--current best practice for relu
    if hp.initializer == "xavier"
        for l = 2:nnp.output_layer
            push!(nnp.theta, randn(nnp.theta_dims[l]) .* sqrt(2.0/nnp.theta_dims[l][2])) # sqrt of no. of input units
        end
    else
        for l = 2:nnp.output_layer
            push!(nnp.theta, zeros(nnp.theta_dims[l])) # sqrt of no. of input units
        end
    end

    # bias initialization: random non-zero initialization performs worse
    nnp.bias = [zeros(size(th, 1)) for th in nnp.theta]  # initialize biases to zero

    # structure of gradient matches theta
    nnp.delta_w = deepcopy(nnp.theta)
    nnp.delta_b = deepcopy(nnp.bias)

    # initialize gradient, 2nd order gradient for Momentum or Adam
    if hp.opt == "momentum" || hp.opt == "adam"
        nnp.delta_v_w = [zeros(size(a)) for a in nnp.delta_w]
        nnp.delta_v_b = [zeros(size(a)) for a in nnp.delta_b]
    end
    if hp.opt == "adam"
        nnp.delta_s_w = [zeros(size(a)) for a in nnp.delta_w]
        nnp.delta_s_b = [zeros(size(a)) for a in nnp.delta_b]
    end

end


"""
    Pre-allocate these arrays for the training batch--either minibatches or one big batch
    Arrays: epsilon, grad, delta_z_norm, delta_z, dropout_random, dropout_mask_units


    NOT USED  -- PROBABLY WON'T WORK AS IS WHEN GOING BACK TO SLICE APPROACH INSTEAD OF VIEW APPROACH

"""
function preallocate_minibatch!(mb, nnp, hp)

    mb.epsilon = [zeros(nnp.layer_units[l], hp.mb_size) for l in 1:nnp.output_layer]
    mb.grad = deepcopy(mb.epsilon)   

    if hp.do_batch_norm
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


function preallocate_batchnorm!(bn, mb, layer_units)
    # initialize batch normalization parameters gamma and beta
    # vector at each layer corresponding to no. of inputs from preceding layer, roughly "features"
    # gamma = scaling factor for normalization standard deviation
    # beta = bias, or new mean instead of zero
    # should batch normalize for relu, can do for other unit functions
    # note: beta and gamma are reserved keywords, using bet and gam
    bn.gam = [ones(i) for i in layer_units]  # gamma is a builtin function
    bn.bet = [zeros(i) for i in layer_units] # beta is a builtin function
    bn.delta_gam = [zeros(i) for i in layer_units]
    bn.delta_bet = [zeros(i) for i in layer_units]
    bn.mu = [zeros(i) for i in layer_units]  # same size as bias = no. of layer units
    bn.mu_run = [zeros(i) for i in layer_units]
    bn.stddev = [zeros(i) for i in layer_units]
    bn.std_run = [zeros(i) for i in layer_units]

end


"""
define and choose functions to be used in neural net training
"""
function setup_functions!(hp, train)
    !hp.quiet && println("Setup functions beginning")
    # make these function variables module level 
        # the layer functions they point to are all module level (in file layer_functions.jl)
        # these are just substitute names or aliases
        # don't freak out about the word global
    global unit_function!
    global gradient_function!
    global classify_function!
    global batch_norm_fwd!
    global batch_norm_back!
    global cost_function
    global optimization_function!

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

    optimization_function! = 
        if hp.opt == "momentum"
            momentum!
        elseif hp.opt == "adam"
            adam!
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
    if !hp.dobatch # no batches--must plot by epoch
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

    plot_labels = [pl for pl in keys(plot_switch) if plot_switch[pl] == true &&
        (pl != "learning" && pl != "cost" && pl != "epoch" && pl != "batch")]  # Cost, Learning are separate plots, not series labels
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

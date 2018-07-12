
"""
function extract_data(matfname::String, norm_mode::String="none")

Extract data from a matlab formatted binary file.

The matlab file may contain these keys: 
- "train", which is required, 
- "test", which is optional.

Within each top-level key the following keys must be present:
- "x", which holds the examples in variables as columns (no column headers should be used)
- "y", which holds the labels as columns for each example (it is possible to have multiple output columns for categories)

Multiple Returns:
-    inputs..........2d array of float64 with rows as features and columns as examples
-    targets.........2d array of float64 with columns as examples
-    test_inputs.....2d array of float64 with rows as features and columns as examples
-    test_targets....2d array of float64 with columns as examples
-    norm_factors.....1d vector of float64 containing [x_mu, x_std] or [m_min, x_max]
.....................note that the factors may be vectors for multiple rows of x

"""
function extract_data(matfname::String)
    # read the data
    df = matread(matfname)

    # Split into train and test datasets, if we have them
    # transpose examples as columns to optimize for julia column-dominant operations
    # e.g.:  rows of a single column are features; each column is an example data point
    if in("train", keys(df))
        inputs = df["train"]["x"]'
        targets = df["train"]["y"]'
    else
        inputs = df["x"]'
        targets = df["y"]'
    end
    if in("test", keys(df))
        test_inputs = df["test"]["x"]'  # note transpose operator
        test_targets = df["test"]["y"]'
    else
        test_inputs = zeros(0,0)
        test_targets = zeros(0,0)
    end

    return inputs, targets, test_inputs, test_targets
end


function normalize_inputs(inputs, test_inputs, norm_mode="none")
    if lowercase(norm_mode) == "standard"
        # normalize training data
        x_mu = mean(inputs, 2)
        x_std = std(inputs, 2)
        inputs = (inputs .- x_mu) ./ (x_std .+ 1e-08)
        # normalize test data
        if size(test_inputs) != (0,0)
            test_inputs = (test_inputs .- x_mu) ./ (x_std .+ 1e-08)
        end
        norm_factors = (x_mu, x_std) # tuple of Array{Float64,2}
    elseif lowercase(norm_mode) == "minmax"
        # normalize training data
        x_max = maximum(inputs, 2)
        x_min = minimum(inputs, 2)
        inputs = (inputs .- x_min) ./ (x_max .- x_min .+ 1e-08)
        # normalize test data
        if size(test_inputs) != (0,0)
            test_inputs = (test_inputs .- x_min) ./ (x_max .- x_min .+ 1e-08)
        end
        norm_factors = (x_min, x_max) # tuple of Array{Float64,2}
    else  # handles case of "", "none" or really any crazy string
        norm_factors = ([0.0], [1.0])
    end

    # to translate to unnormalized regression coefficients: m = mhat / stdx, b = bhat - (m*xmu)
    # precalculate a and b constants, and 
    # then just apply newvalue = a * value + b. a = (max'-min')/(max-min) and b = max - a * max 
    # (x - x.min()) / (x.max() - x.min())       # values from 0 to 1
    # 2*(x - x.min()) / (x.max() - x.min()) - 1 # values from -1 to 1

    return inputs, test_inputs, norm_factors
end


function normalize_replay!(inputs, norm_mode, norm_factors)
    if norm_mode == "standard"
        x_mu = norm_factors[1]
        x_std = norm_factors[2]
        inputs = (inputs .- x_mu) ./ (x_std .+ 1e-08)
    elseif norm_mode == "minmax"
        x_min = norm_factors[1]
        x_max = norm_factors[2]
        inputs = (inputs .- x_min) ./ (x_max .- x_min .+ 1e-08)
    else
        error("Input norm_mode = $norm_mode must be standard or minmax")
    end
end


function setup_model!(mb, hp, tp, bn, dotest, train, test)

    #setup mini-batch
    if hp.mb_size_in < 1
        hp.mb_size_in = train.n  # use 1 (mini-)batch with all of the examples
        hp.mb_size = train.n
    elseif hp.mb_size_in >= train.n
        hp.mb_size_in = train.n
        hp.mb_size = train_n
    else 
        hp.mb_size = hp.mb_size_in
    end
    hp.n_mb = ceil(Int, train.n / hp.mb_size)  # number of mini-batches
    hp.alphaovermb = hp.alpha / hp.mb_size  # calc once, use in hot loop
    hp.do_batch_norm = hp.n_mb == 1 ? false : hp.do_batch_norm  # no batch normalization for 1 batch

    # randomize order of all training samples:
        # labels in training data often in a block, which will make
        # mini-batch train badly because a batch will not contain mix of target labels
    if hp.mb_size < train.n
        select_index = randperm(train.n)
        train.inputs[:] = train.inputs[:, select_index]
        train.targets[:] = train.targets[:, select_index]
    end

    # set parameters for Momentum or Adam optimization
    if hp.opt == "momentum" || hp.opt == "adam"
        if !(hp.opt_params == [])  # use inputs for opt_params
            # set b1 for Momentum and Adam
            if hp.opt_params[1] > 1.0 || hp.opt_params[1] < 0.5
                warn("First opt_params for momentum or adam should be between 0.5 and 0.999. Using default")
                # nothing to do:  hp.b1 = 0.9 and hp.b2 = 0.999 and hp.ltl_eps = 1e-8
            else
                hp.b1 = hp.opt_params[1] # use the passed parameter
            end
            # set b2 for Adam
            if length(hp.opt_params) > 1
                if hp.opt_params[2] > 1.0 || hp.opt_params[2] < 0.9
                    warn("second opt_params for adam should be between 0.9 and 0.999. Using default")
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
    #    drop_ran_w and drop_filt_w are in mb or train (Model_data)
    # set a droplim for each layer (input layer and output layer will be ignored)
    if hp.dropout
        # fill droplim to match number of hidden layers
        if length(hp.droplim) > length(hp.n_hid)
            hp.droplim = hp.droplim[1:length(hp.n_hid)] # truncate
        elseif length(hp.droplim) < length(hp.n_hid)
            for i = 1:length(hp.n_hid)-length(hp.droplim)
                push!(hp.droplim,hp.droplim[end]) # pad
            end
        end
        hp.droplim = [1.0, hp.droplim..., 1.0] # placeholders for input and output layers
    end

    # debug
    # println("opt params: $(hp.b1), $(hp.b2)")

    # DEBUG see if hyper_parameters set correctly
    # for sym in fieldnames(hp)
    #    println(sym," ",getfield(hp,sym), " ", typeof(getfield(hp,sym)))
    # end

    ##########################################################################
    #  pre-allocate data storage
    ##########################################################################

    preallocate_nn_params!(tp, hp, hp.n_hid, train.in_k, train.n, train.out_k)

    preallocate_data!(train, tp, train.n, hp)

    # feedfwd test data--if test input found
    if dotest
        test.n = size(test.inputs,2)
        preallocate_data!(test, tp, test.n, hp)
    else
        test.n = 0
    end

    # pre-allocate feedfwd mini-batch training data
    # preallocate_minibatch!(mb, tp, hp)  
    
    # batch normalization parameters
    if hp.do_batch_norm
        preallocate_batchnorm!(bn, mb, tp.layer_units)
    end

    # debug
    # verify correct dimensions of dropout filter
    # for item in mb.drop_filt_w
    #     println(size(item))
    # end
    # error("that's all folks!....")

end


####################################################################
#  functions to pre-allocate data updated during training loop
####################################################################

# use for test and training data
function preallocate_data!(dat, nnp, n, hp)
    # feedforward
    dat.a = [dat.inputs]
    dat.z = [zeros(size(dat.inputs))] # not used for input layer  TODO--this permeates the code but not needed
    for i = 2:nnp.output_layer-1  # hidden layers
        push!(dat.z, zeros(nnp.layer_units[i], n))
        push!(dat.a, zeros(size(dat.z[i])))  #  and up...  ...output layer set after loop
    end
    push!(dat.z, zeros(size(nnp.theta[nnp.output_layer],1),n))
    push!(dat.a, zeros(size(nnp.theta[nnp.output_layer],1),n))

    # training / backprop  -- pre-allocate only minibatch size (except last one, which could be smaller)
    dat.epsilon = [i[:,1:hp.mb_size_in] for i in dat.a]
    dat.grad = [i[:,1:hp.mb_size_in] for i in dat.a]
    dat.delta_z = [i[:,1:hp.mb_size_in] for i in dat.a]

    if hp.do_batch_norm  # required for full pass performance stats
        # feedforward
        dat.z_norm = deepcopy(dat.z)
        # backprop
        dat.delta_z_norm = [i[:,1:hp.mb_size_in] for i in dat.a]
    end

    # backprop / training
    if hp.dropout
        dat.drop_ran_w = [i[:,1:hp.mb_size_in] for i in dat.a]
        dat.drop_filt_w = [BitArray(ones(size(i,1),hp.mb_size_in)) for i in dat.a]
    end

end


function preallocate_nn_params!(tp, hp, n_hid, in_k, n, out_k)
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
    tp.output_layer = 2 + size(n_hid, 1) # input layer is 1, output layer is highest value
    tp.layer_units = [in_k, n_hid..., out_k]

    # set dimensions of the linear weights for each layer
    push!(tp.theta_dims, (in_k, 1)) # weight dimensions for the input layer -- if using array, must splat as arg
    for l = 2:tp.output_layer-1  # l refers to nn layer so this includes only hidden layers
        push!(tp.theta_dims, (tp.layer_units[l], tp.layer_units[l-1]))
    end
    push!(tp.theta_dims, (out_k, tp.layer_units[tp.output_layer - 1]))  # weight dims for output layer: rows = output classes

    # initialize the linear weights
    tp.theta = [zeros(2,2)] # layer 1 not used

    # Xavier initialization--current best practice for relu
    for l = 2:tp.output_layer
        push!(tp.theta, randn(tp.theta_dims[l]) .* sqrt(2.0/tp.theta_dims[l][2])) # sqrt of no. of input units
    end

    # bias initialization: random non-zero initialization performs worse
    tp.bias = [zeros(size(th, 1)) for th in tp.theta]  # initialize biases to zero

    # structure of gradient matches theta
    tp.delta_w = deepcopy(tp.theta)
    tp.delta_b = deepcopy(tp.bias)

    # initialize gradient, 2nd order gradient for Momentum or Adam
    if hp.opt == "momentum" || hp.opt == "adam"
        tp.delta_v_w = [zeros(size(a)) for a in tp.delta_w]
        tp.delta_v_b = [zeros(size(a)) for a in tp.delta_b]
    end
    if hp.opt == "adam"
        tp.delta_s_w = [zeros(size(a)) for a in tp.delta_w]
        tp.delta_s_b = [zeros(size(a)) for a in tp.delta_b]
    end

end


"""
    Pre-allocate these arrays for the training batch--either minibatches or one big batch
    Arrays: epsilon, grad, delta_z_norm, delta_z, drop_ran_w, drop_filt_w


    NOT USED  -- PROBABLY WON'T WORK AS IS WHEN GOING BACK TO SLICE APPROACH INSTEAD OF VIEW APPROACH

"""
function preallocate_minibatch!(mb, tp, hp)

    mb.epsilon = [zeros(tp.layer_units[l], hp.mb_size) for l in 1:tp.output_layer]
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
        mb.drop_ran_w = deepcopy(mb.epsilon)
        push!(mb.drop_filt_w,fill(true,(2,2))) # for input layer, not used
        for item in mb.drop_ran_w[2:end]
            push!(mb.drop_filt_w,fill(true,size(item)))
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
function setup_functions!(units, out_k, opt, classify)

    # all the other functions are module level, e.g. global--make these function variables module level, too
    global unit_function!
    global gradient_function!
    global classify_function!
    global batch_norm_fwd!
    global batch_norm_back!
    global cost_function
    global optimization_function!

    unit_function! =
        if units == "sigmoid"
            sigmoid!
        elseif units == "l_relu"
            l_relu!
        elseif units == "relu"
            relu!
        elseif units == "tanh"
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
        if out_k > 1  # more than one output (unit)
            if classify == "sigmoid"
                sigmoid!
            elseif classify == "softmax"
                softmax!
            else
                error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
            end
        else
            if classify == "sigmoid"
                sigmoid!  # for one output label
            elseif classify == "regression"
                regression!
            else
                error("Function to classify output must be \"sigmoid\" or \"regression\".")
            end
        end

    optimization_function! = 
        if opt == "momentum"
            momentum!
        elseif opt == "adam"
            adam!
        else
            no_optimization
        end

    # set cost function
    cost_function = 
        if classify=="regression" 
            mse_cost 
        else
            cross_entropy_cost
        end

end


"""
Function setup_plots(epochs::Int64, dotest::Bool, plots::Array{String,1})

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
    "fracright_history"=>fracright_history: an array of percentage of correct classification
        at each iteration with iterations as rows and result types as columns ("Training", "Test").
        This plots a so-called learning curve.  Very interesting indeed.
    "col_train"=>col_train: column of the arrays above to be used for Training results
    "col_test"=>col_test: column of the arrays above to be used for Test results

"""
function setup_plots(epochs::Int64, dotest::Bool, plots::Array{String,1})
    # set up cost_history to track 1 or 2 data series for plots
    # lots of indirection here:  someday might add "validation"
    if size(plots,1) > 4
        warn("Only 4 plot requests permitted. Proceeding with up to 4.")
    end

    valid_plots = ["Training", "Test", "Learning", "Cost"]
    if in(plots, ["None", "none", ""])
        plot_switch = Dict(pl => false for pl in valid_plots) # set all to false
    else
        plot_switch = Dict(pl => in(pl, plots) for pl in valid_plots)
    end

    # must have test data to plot test results
    if dotest  # test data is present
        # nothing to change
    else
        if plot_switch["Test"]  # input requested plotting test data results
            warn("Can't plot test data. No test data. Proceeding.")
            plot_switch["Test"] = false
        end
    end

    plot_labels = [pl for pl in keys(plot_switch) if plot_switch[pl] == true &&
        (pl != "Learning" && pl != "Cost")]  # Cost, Learning are separate plots, not series labels
    plot_labels = reshape(plot_labels,1,size(plot_labels,1)) # 1 x N row array required by pyplot

    plotdef = Dict("plot_switch"=>plot_switch, "plot_labels"=>plot_labels)

    if plot_switch["Cost"]
        cost_history = zeros(epochs, size(plot_labels,2))
        plotdef["cost_history"] = cost_history
    end
    if plot_switch["Learning"]
        fracright_history = zeros(epochs, size(plot_labels,2))
        plotdef["fracright_history"] = fracright_history
    end

    # set column in cost_history for each data series
    col_train = plot_switch["Training"] ? 1 : 0
    col_test = plot_switch["Test"] ? col_train + 1 : 0

    plotdef["col_train"] = col_train
    plotdef["col_test"] = col_test

    return plotdef
end

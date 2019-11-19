using Plots
using JLD2
using Printf
using LinearAlgebra


function prep_training!(mb, hp, nnw, bn, train)
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

    hp.do_learn_decay && (hp.learn_decay = [hp.learn_decay[1], floor(hp.learn_decay[2])])

    # dropout parameters: droplim is in hp (Hyper_parameters),
    #    dropout_random and dropout_mask_units are in mb or train (Model_data)
    # set a droplim for each layer 
    if hp.dropout
        if length(hp.droplim) == length(hp.hidden) + 2
            # droplim supplied for every layer
            if hp.droplim[end] != 1.0
                @warn("Poor performance when dropping units from output layer, continuing.")
            end
        elseif length(hp.droplim) == length(hp.hidden) + 1
            # droplim supplied for input layer and hidden layers
            hp.droplim = [hp.droplim..., 1.0]  # keep all units in output layer
        elseif length(hp.droplim) < length(hp.hidden)
            # pad to provide same droplim for all hidden layers
            for i = 1:length(hp.hidden)-length(hp.droplim)
                push!(hp.droplim,hp.droplim[end]) 
            end
            hp.droplim = [1.0, hp.droplim..., 1.0] # use all units for input and output layers
        else
            @warn("More drop limits provided than total network layers, use limits ONLY for hidden layers.")
            hp.droplim = hp.droplim[1:length(hp.hidden)]  # truncate
            hp.droplim = [1.0, hp.droplim..., 1.0] # placeholders for input and output layers
        end
    end

    # setup parameters for maxnorm regularization
    if titlecase(hp.reg) == "Maxnorm"
        hp.reg = "Maxnorm"
        if isempty(hp.maxnorm_lim)
            @warn("Values in Float64 array must be set for maxnormlim to use Maxnorm Reg, continuing without...")
            hp.reg = "L2"
        elseif length(hp.maxnorm_lim) == length(hp.hidden) + 1
            hp.maxnorm_lim = append!([0.0], hp.maxnorm_lim) # add dummy for input layer
        elseif length(hp.maxnorm_lim) > length(hp.hidden) + 1
            @warn("Too many values in maxnorm_lim; truncating to hidden and output layers.")
            hp.maxnorm_lim = append!([0.0], hp.maxnorm_lim[1:length(hp.hidden)+1]) # truncate and add dummy for input layer
        elseif length(hp.maxnorm_lim) < length(hp.hidden) + 1
            for i = 1:length(hp.hidden)-length(hp.maxnorm_lim) + 1
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




"""
define and choose functions to be used in neural net training
"""
function setup_functions!(hp, train)
    # assumes hp has been verified--no error checking here!

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
    global dropout_fwd_function!
    global dropout_back_function!

    n_layers = length(hp.hidden) + 2

    # unitfunctions = Dict("sigmoid" => sigmoid!, "l_relu" => l_relu!, "relu" => relu!, "tanh" => tanh_act!)
    # gradientfunctions = Dict("sigmoid" => sigmoid_gradient!, "l_relu" => l_relu_gradient!, 
    #                         "relu" => relu_gradient!, "tanh" => tanh_act_gradient!)
    # classifyfunctions = Dict("sigmoid" => sigmoid!, "softmax" => softmax!)
    # regfunctions = Dict("L2" => l2_reg!, "L1" => l1_reg!, "Maxnorm" => maxnorm_reg!, "" => no_reg, "none" => no_reg)
    # optimizationfunctions = Dict("momentum" => momentum!, "adam" => adam!, "rmsprop" => rmsprop!, "" => no_optimization,
    #                      "none" => no_optimization)
    # costfunctions = Dict("regression" => mse_cost, "default" => cross_entropy_cost)

    # unit_function! = get(unitfunctions, hp.units, sigmoid!)
    # gradient_function! = get(gradientfunctions, hp.units, sigmoid!)
    # classify_function! = get(classifyfunctions, hp.classify, sigmoid!)
    # # check outputs
    # reg_function! = get(regfunctions, hp.reg, noop)
    # optimization_function! = get(optimizationfunctions, hp.opt, noop)
    # cost_function = get(costfunctions, hp.classify_function, cross_entropy_cost)

    # allow different functions at each appropriate layer
    unit_function! = Array{Function}(undef, n_layers)
    gradient_function! = Array{Function}(undef, n_layers)
    reg_function! = Array{Function}(undef, n_layers)
    dropout_fwd_function! = Array{Function}(undef, n_layers)
    dropout_back_function! = Array{Function}(undef, n_layers)

    for layer in 2:n_layers-1 # for hidden layers: layers 2 through output - 1
        hidden_layer = layer - 1
        unit_function![layer] =
            if hp.hidden[hidden_layer][1] == "sigmoid"
                sigmoid!
            elseif hp.hidden[hidden_layer][1] == "l_relu"
                l_relu!
            elseif hp.hidden[hidden_layer][1] == "relu"
                relu!
            elseif hp.hidden[hidden_layer][1] == "tanh"
                tanh_act!
            end

        gradient_function![layer] =
            if unit_function![layer] == sigmoid! # i + 1 is walks through the hidden layers
                sigmoid_gradient!
            elseif unit_function![layer] == l_relu!
                l_relu_gradient!
            elseif unit_function![layer] == relu!
                relu_gradient!
            elseif unit_function![layer] == tanh_act!
                tanh_act_gradient!
            end

        dropout_fwd_function![layer] = 
            if hp.dropout && (hp.droplim[hl] < 1.0)
                dropout_fwd!
            else
                noop
            end

        dropout_back_function![layer] =
            if hp.dropout && (hp.droplim[hl] < 1.0)
                dropout_back!
            else
                noop
            end

    end

    # TODO update to enable different regulization at each layer
    for layer = 2:n_layers  # from the first hidden layer=2 to output layer
        reg_function![layer] = 
            if hp.reg == "L2"
                l2_reg!
            elseif hp.reg == "L1"
                l1_reg!
            elseif hp.reg == "Maxnorm"
                maxnorm_reg!
            else
                noop
            end
        
    end

    optimization_function! = 
        if hp.opt == "momentum"
            momentum!
        elseif hp.opt == "adam"
            adam!
        elseif hp.opt == "rmsprop"
            rmsprop!
        else
            noop
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

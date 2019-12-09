# using Plots
using JLD2
using Printf
using LinearAlgebra


function prep_training!(mb, hp, nnw, bn, n)
    !hp.quiet && println("Setup_model beginning")
    !hp.quiet && println("hp.dobatch: ", hp.dobatch)

    # debug
    # println("norm_factors ", typeof(norm_factors))
    # println(norm_factors)

    #setup mini-batch
    if hp.dobatch
        @info("Be sure to shuffle training data when using minibatches.  Use utility function shuffle_data! or your own.")
        if hp.mb_size_in < 1
            hp.mb_size = hp.mb_size_in = n  
            hp.dobatch = false    # user provided incompatible inputs
        elseif hp.mb_size_in >= n
            hp.mb_size = hp.mb_size_in = n
            hp.dobatch = false   # user provided incompatible inputs
        else 
            hp.mb_size = hp.mb_size_in
        end
        hp.alphaovermb = hp.alpha / hp.mb_size  # calc once, use in hot loop
        hp.do_batch_norm = hp.dobatch ? hp.do_batch_norm : false  
    else
        hp.alphaovermb = hp.alpha / n 
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
function setup_functions!(hp, nnw, bn, dat)
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
    global affine_function!
    global batch_norm_fwd_function!
    global batch_norm_fwd_predict_function!
    global batch_norm_back_function!
    global backprop_weights_function!

    n_layers = length(hp.hidden) + 2

    # allow different functions at each appropriate layer
    unit_function! = Array{Function}(undef, n_layers)
    gradient_function! = Array{Function}(undef, n_layers)
    reg_function! = Array{Function}(undef, n_layers)
    dropout_fwd_function! = Array{Function}(undef, n_layers)
    dropout_back_function! = Array{Function}(undef, n_layers)

    for layer in 2:n_layers-1 # for hidden layers: layers 2 through output - 1
        hidden_layer = layer - 1
        unit_function![layer] =  # hidden looks like [["relu",100], ...]
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
            if unit_function![layer] == sigmoid! 
                sigmoid_gradient!
            elseif unit_function![layer] == l_relu!
                l_relu_gradient!
            elseif unit_function![layer] == relu!
                relu_gradient!
            elseif unit_function![layer] == tanh_act!
                tanh_act_gradient!
            end

        dropout_back_function![layer] =
            if hp.dropout && (hp.droplim[hl] < 1.0)
                dropout_back!
            else
                noop
            end
    end

    for layer = 1:n_layers-1 # input layer and hidden layers
        dropout_fwd_function![layer] = 
            if hp.dropout && (hp.droplim[hl] < 1.0)
                dropout_fwd!
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
        if dat.out_k > 1  # more than one output (unit)
            if hp.classify == "sigmoid"
                sigmoid!
            elseif hp.classify == "softmax"
                softmax!
            else
                error("Function to classify multiple output labels must be \"sigmoid\" or \"softmax\".")
            end
        else
            if hp.classify == "sigmoid" || hp.classify == "logistic"
                logistic!  # for one output label
            elseif hp.classify == "regression"
                regression!
            else
                error("Function to classify single output must be \"sigmoid\", \"logistic\" or \"regression\".")
            end
        end

    cost_function = 
        if hp.classify=="regression" 
            mse_cost 
        else
            cross_entropy_cost
        end

    affine_function! = 
        if hp.do_batch_norm
            affine_nobias!  # same arguments, just ignores bias in calculation
        else
            affine!
        end

    batch_norm_fwd_function! = 
        if hp.do_batch_norm
            batch_norm_fwd_function! = create_curry_batch_norm_fwd!(hp, bn)
        else
            noop
        end

    batch_norm_fwd_predict_function! = 
        if hp.do_batch_norm
            batch_norm_fwd_predict_function! = create_curry_batch_norm_fwd_predict!(hp, bn)
        else
            noop
        end

    batch_norm_back_function! =
        if hp.do_batch_norm
            batch_norm_back_function! = create_curry_batch_norm_back!(hp, bn, nnw)
        else
            noop
        end

    backprop_weights_function! = 
        if hp.do_batch_norm
            backprop_weights_nobias!
        else
            backprop_weights!
        end

    !hp.quiet && println("Setup functions completed.")
end

##############################################################
# curried functions used by setup_functions
##############################################################

function create_curry_batch_norm_fwd!(hp, bn)  # arguments that are captured and "built-in" to curried function
    return function f(dat, hl)  # arguments that will be passed in when new function called
                batch_norm_fwd!(hp, bn, dat, hl)
           end  # actual name of the resulting function set to return result of create_curry function
end


function create_curry_batch_norm_fwd_predict!(hp, bn)  # arguments that are captured and "built-in" to curried function
    return function f(dat, hl)  # arguments that will be passed in when new function called
                batch_norm_fwd_predict!(hp, bn, dat, hl)
           end  # actual name of the resulting function set to return result of create_curry function
end


function create_curry_batch_norm_back!(hp, bn, nnw)  # arguments that are captured and "built-in" to curried function
    return function f(dat, hl)  # arguments that will be passed in when new function called
                batch_norm_back!(nnw, dat, bn, hl, hp)
           end  # actual name of the resulting function set to return result of create_curry function
end



"""
Function setup_stats(hp, dotest::Bool)

Creates data structure to hold everything needed to plot progress of
neural net training by iteration.

A statsdat is a dict containing:

    "stats_sel"=>stats_sel: Dict of bools to select each type of results to be collected.
        Currently used are: "Training", "Test", "Learning", "Cost".  This determines what
        data will be collected during training iterations and what data series will be
        plotted.
    "stats_labels"=>stats_labels: array of strings provides the labels to be used in the
        plot legend.
    "cost_history"=>cost_history: an array of calculated cost at each iteration
        with iterations as rows and result types ("Training", "Test") as columns.
    "accuracy"=>accuracy: an array of percentage of correct classification
        at each iteration with iterations as rows and result types as columns ("Training", "Test").
        This plots a so-called learning curve.  Very interesting indeed.
    "col_train"=>col_train: column of the arrays above to be used for Training results
    "col_test"=>col_test: column of the arrays above to be used for Test results

"""
function setup_stats(hp, dotest::Bool)
    # set up cost_history to track 1 or 2 data series for plots
    # lots of indirection here:  someday might add "validation"
    if size(hp.stats,1) > 5
        @warn("Only 4 plot requests permitted. Proceeding with up to 4.")
    end

    valid_stats = ["train", "test", "learning", "cost", "epoch", "batch"]
    if in(hp.stats, ["None", "none", ""])
        stats_sel = Dict(pl => false for pl in valid_stats) # set all to false
    else
        stats_sel = Dict(pl => in(pl, hp.stats) for pl in valid_stats)
    end

    # determine whether to plot per batch or per epoch
    if in(hp.stats, ["None", "none", ""])
        hp.plotperbatch = false
        hp.plotperepoch = false        
    elseif !hp.dobatch # no batches--must plot by epoch
        hp.plotperbatch = false
        hp.plotperepoch = true
    elseif in("epoch", hp.stats) # this is the default and overrides conflicting choice
        hp.plotperbatch = false
        hp.plotperepoch = true
    elseif in("batch", hp.stats)
        hp.plotperbatch = true
        hp.plotperepoch = false
    else # even when NOT plotting we still gather stats in the statsdat for default epoch
        hp.plotperbatch = false
        hp.plotperepoch = true
    end

    pointcnt = hp.plotperepoch ? hp.epochs : hp.n_mb * hp.epochs

    # must have test data to plot test results
    if !dotest  # no test data
        if stats_sel["test"]  # input requested plotting test data results
            @warn("Can't plot test data. No test data. Proceeding.")
            stats_sel["test"] = false
        end
    end

    stats_labels = dotest ? ["Train", "Test"] : ["Train"]
    stats_labels = reshape(stats_labels,1,size(stats_labels,1)) # 1 x N row array required by pyplot

    statsdat = Dict("stats_sel"=>stats_sel, "stats_labels"=>stats_labels)

    if stats_sel["cost"]
        statsdat["cost_history"] = zeros(pointcnt, size(stats_labels,2)) # cost history initialized to 0's
    end
    if stats_sel["learning"]
        statsdat["accuracy"] = zeros(pointcnt, size(stats_labels,2))
    end

    # set column in cost_history for each data series
    col_train = stats_sel["train"] ? 1 : 0
    col_test = stats_sel["test"] ? col_train + 1 : 0

    statsdat["train"] = col_train
    statsdat["test"] = col_test

    return statsdat
end

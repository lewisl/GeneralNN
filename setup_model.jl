# using Plots
using JLD2
using Printf
using LinearAlgebra


function create_model!(model, hp, nnw)
!hp.quiet && println("Create model beginning")

    func_dict = create_funcs() # for both feed forward and back propagation
    model.ff_strstack = build_ff_string_stack(hp, nnw)
    model.ff_execstack = build_exec_stack(model.ff_strstack, func_dict)
    model.back_strstack = build_back_string_stack(hp)
    model.back_execstack = build_exec_stack(model.back_strstack, func_dict)
    model.update_strstack = build_update_string_stack(hp)
    model.update_execstack = build_exec_stack(model.update_strstack, func_dict)

    model.cost_function = setup_functions!(hp)

    !hp.quiet && println("Create model completed")
end


function prep_training!(mb, hp, nnw, bn, n)
    !hp.quiet && println("prep training beginning")
    !hp.quiet && println("hp.dobatch: ", hp.dobatch)

    # debug
    # println("norm_factors ", typeof(norm_factors))
    # println(norm_factors)

    #setup mini-batch
    if hp.dobatch
        @info("Be sure to shuffle training data when using minibatches.\n  Use utility function shuffle_data! or your own.")
        if hp.mb_size_in < 1
            hp.mb_size_in = n  
            hp.dobatch = false    # user provided incompatible inputs
        elseif hp.mb_size_in >= n
            hp.mb_size_in = n
            hp.dobatch = false   # user provided incompatible inputs
        end 
        hp.mb_size = hp.mb_size_in  # start value for hp.mb_size; changes if last minibatch is smaller
        hp.do_batch_norm = hp.dobatch ? hp.do_batch_norm : false  
    else
        hp.mb_size_in = n
        hp.mb_size = float(n)
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
        if length(hp.droplim) == length(hp.hidden) + 2  # droplim for every layer
            if hp.droplim[end] != 1.0
                @warn("Poor performance when dropping units from output layer, continuing.")
            end
        elseif length(hp.droplim) == length(hp.hidden) + 1 # droplim for input and hidden layers
            hp.droplim = [hp.droplim..., 1.0]  # keep all units in output layer
        elseif length(hp.droplim) < length(hp.hidden)  # pad droplim for all hidden layers
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
    struct MBrng  # values are set once to define the iterator stop point=cnt, and increment=incr
        cnt::Int
        incr::Int
    end

    function mbiter(mb::MBrng, start)  # new method for Base.iterate
        nxtstart = start + mb.incr
        stop = nxtstart - 1 < mb.cnt ? nxtstart - 1 : mb.cnt
        ret = start < mb.cnt ? (start:stop, nxtstart) : nothing # return tuple of range and next state, or nothing--to stop iteration
        return ret
    end

    function mblength(mb::MBrng)  # new method for Base.length
        return ceil(Int,mb.cnt / mb.incr)
    end

    # add iterate methods: must supply type for the new methods--method dispatch selects the method for this type of iterator
        # the function  names don't matter--we provide an alternate for the standard methods, but the functions
        # need to do the right things
    Base.iterate(mb::MBrng, start=1) = mbiter(mb::MBrng, start)   # canonical to use "state" instead of "start"
    Base.length(mb::MBrng) = mblength(mb)


function build_ff_string_stack(hp, out_k)
    strstack = []
    n_hid = length(hp.hidden)
    n_layers = n_hid + 2

    #input layer
    layer = 1
        layer_group = String[]
        if hp.dropout && (hp.droplim[1] < 1.0)
            push!(layer_group, "dropout")
        end
    push!(strstack, layer_group)

    # hidden layers
    for layer = 2:n_hid+1
        layer_group = String[]
        hl = layer - 1

        if hp.dropout && (hp.droplim[1] < 1.0)
            push!(layer_group, "dropout_fwd")
        end

        # affine either or...
        hp.do_batch_norm && push!(layer_group, "affine_nobias") # true
        hp.do_batch_norm || push!(layer_group, "affine") # false
        # batch_norm_fwd 
        hp.do_batch_norm && push!(layer_group, "batch_norm_fwd")
        # unit_function or activation --> updates in place
        unit_function = 
            if hp.hidden[hl][1] == "sigmoid"  # hp.hidden looks like [["relu", 80], ["sigmoid", 80]]
                "sigmoid"
            elseif hp.hidden[hl][1] == "l_relu"
                "l_relu"
            elseif hp.hidden[hl][1] == "relu"
                "relu"
            elseif hp.hidden[hl][1] == "tanh"
                "tanh_act"
            end
        push!(layer_group, unit_function)
        # dropout --> updates in place
        if hp.dropout && (hp.droplim[layer] < 1.0)
            push!(layer_group, "dropout_fwd")
        end        

        # done with layer_group for current hidden layer
        push!(strstack, layer_group) 
    end

    #   output layer
    layer = n_hid + 2
    layer_group = String[]

        push!(layer_group, "affine")
        # classify_function --> updates in place
        classify_function = 
            if out_k > 1  # more than one output label (unit)
                if hp.classify == "sigmoid"
                    "sigmoid"
                elseif hp.classify == "softmax"
                    "softmax"
                else
                    error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
                end
            else
                if hp.classify == "sigmoid" || hp.classify == "logistic"
                    "logistic"  # for one output label
                elseif hp.classify == "regression"
                    "regression"
                else
                    error("Function to classify output must be \"sigmoid\", \"logistic\" or \"regression\".")
                end
            end
        push!(layer_group, classify_function)
    push!(strstack, layer_group) 

    return strstack
end   


function build_exec_stack(strstack, func_dict)
    execstack= []
    for i in 1:size(strstack,1)
        push!(execstack,[])
        for r in strstack[i]
            push!(execstack[i], func_dict[r])
        end
    end

    return execstack
end


function build_back_string_stack(hp)
    strstack = []
    n_hid = length(hp.hidden)
    n_layers = n_hid + 2

    # input layer
    layer = 1
    layer_group = String[]
        # usually nothing to do here  TODO resolve issue about backprop for the weights from the input layer
    push!(strstack, layer_group)

    # hidden layers
    for layer in 2:n_layers-1 # for hidden layers: layers 2 through output - 1
        hl = hidden_layer = layer - 1
        layer_group = String[]  # start a new layer_group

        push!(layer_group, "inbound_epsilon")   # required

        if hp.dropout && (hp.droplim[hl] < 1.0)
            push!(layer_group, "dropout_back")
        end
        
        gradient_function =
            if hp.hidden[hidden_layer][1] == "sigmoid" 
                "sigmoid_gradient"
            elseif hp.hidden[hidden_layer][1] == "l_relu"
                "l_relu_gradient"
            elseif hp.hidden[hidden_layer][1] == "relu"
                "relu_gradient"
            elseif hp.hidden[hidden_layer][1] == "tanh_act"
                "tanh_act_gradient"
            end
            push!(layer_group, gradient_function)

        push!(layer_group, "current_lr_epsilon")        # required
        
        hp.do_batch_norm && push!(layer_group, "batch_norm_back")
        # affine either or...
        hp.do_batch_norm && push!(layer_group, "backprop_weights_nobias") # true
        hp.do_batch_norm || push!(layer_group, "backprop_weights") # false

        # done with layer_group for current hidden layer
        push!(strstack, layer_group) 
    end

    #output layer
    layer_group = String[]
        push!(layer_group, "backprop_classify")
        push!(layer_group, "backprop_weights")
    push!(strstack, layer_group)  

    return strstack
end


function build_update_string_stack(hp)
    strstack = []
    n_hid = length(hp.hidden)
    n_layers = n_hid + 2

    # input layer
    layer = 1
        layer_group = String[]
            # usually nothing to do here  TODO resolve issue about backprop for the weights from the input layer
        push!(strstack, layer_group)

    # hidden layers
    for layer in 2:n_layers # for hidden layers and output layer
        hl = hidden_layer = layer - 1
        layer_group = String[]  # start a new layer_group

        optimization_function = 
            if hp.opt == "momentum"
                "momentum"
            elseif hp.opt == "adam"
                "adam"
            elseif hp.opt == "rmsprop"
                "rmsprop"
            else
                "noop"
            end
            if optimization_function == "noop"
            else
                push!(layer_group, optimization_function)
            end

        # update_wgts either or...
        hp.do_batch_norm && push!(layer_group, "update_wgts_nobias") # true -- don't use bias
        hp.do_batch_norm || push!(layer_group, "update_wgts") # false -- use bias
        hp.do_batch_norm && push!(layer_group, "update_batch_norm")

        reg_function = 
            if hp.reg == "L2"
                "l2_reg"
            elseif hp.reg == "L1"
                "l1_reg"
            elseif hp.reg == "Maxnorm"
                "maxnorm_reg"
            else
                "noop"
            end        
            if reg_function == "noop"
            else
                push!(layer_group, reg_function)
            end


        # done with layer_group for current hidden layer
        push!(strstack, layer_group) 
    end

    #output layer
        # same as hidden layers:  included in loop above
        if !hp.opt_output   # false: remove the optimization function from output layer_group
            # optimization works poorly on the output layer with softmax
            optfunc = filter(x->in(x,["adam","rmsprop","momentum"]), strstack[n_layers])
            deleteat!(strstack[n_layers], indexin(optfunc,strstack[n_layers])) 
        end

    return strstack
end


"""
define and choose functions to be used in neural net training
"""
function setup_functions!(hp)  # , nnw, bn, dat

    n_layers = length(hp.hidden) + 2

    cost_function = 
        if hp.classify=="regression" 
            mse_cost 
        elseif hp.classify == "softmax"
            softmax_cost
        else
            cross_entropy_cost
        end

    return cost_function
end


function create_funcs()
    # indexable function container  TODO start with just feed fwd
    func_dict = Dict(   
                     # activation
                    "affine" => affine!,
                    "affine_nobias" => affine_nobias!,
                    "sigmoid" => sigmoid!,
                    "tanh_act" => tanh_act!,
                    "l_relu" => l_relu!,
                    "relu" => relu!,
                    # classification
                    "softmax" => softmax!,
                    "logistic" => logistic!,
                    "regression" => regression!,

                    # batch norm
                    "batch_norm_fwd" => batch_norm_fwd!,

                    # optimization
                    "dropout_fwd" => dropout_fwd!,

                    # back propagation
                    "backprop_classify" => backprop_classify!,
                    "backprop_weights" => backprop_weights!,
                    "backprop_weights_nobias" => backprop_weights_nobias!,

                    "inbound_epsilon" => inbound_epsilon!,
                    "current_lr_epsilon" => current_lr_epsilon!,

                    # gradient
                    "affine_gradient" => affine_gradient!,
                    "sigmoid_gradient" => sigmoid_gradient!,
                    "tanh_act_gradient" => l_relu_gradient!,
                    "l_relu_gradient" => l_relu_gradient!,
                    "relu_gradient" => relu_gradient!,

                    # batch norm
                    "batch_norm_back" => batch_norm_back!,

                    # optimization
                    "dropout_back" => dropout_back!,
                    "momentum" => momentum!,
                    "adam" => adam!,
                    "rmsprop" => rmsprop!,

                    # update parameters
                    "update_wgts" => update_wgts!,
                    "update_wgts_nobias" => update_wgts_nobias!,
                    "update_batch_norm" => update_batch_norm!,

                    # regularization
                    "maxnorm_reg" => maxnorm_reg!,
                    "l1_reg" => l1_reg!,
                    "l2_reg" => l2_reg!
                )
end



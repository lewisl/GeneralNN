# using Plots
using JLD2
using Printf
using LinearAlgebra

"""
    create_model!(model, hp, nnw)

Create training executation stacks for feed forward, back propagation, and updating parameters. A string
version is created for printing and for selecting functions. A function version is created that will be 
executed in the training loop. All are stored in a Model_def struct:

- feedfwd: model.ff_strstack, model.ff_execstack
- backprop: model.back_strstack, model.back_execstack
- update parameters: model.update_strstack, model.update_execstack

Updates the model struct in place.

"""
function create_model!(model, hp, out_k)
!hp.quiet && println("Create model beginning")

    func_dict = create_funcs() # for feed forward, back propagation and update parameters

    model.ff_strstack = build_ff_string_stack(hp, out_k) # string names of functions
    model.ff_execstack = build_exec_stack(model.ff_strstack, func_dict) # executable function references

    model.back_strstack = build_back_string_stack(hp) # string names
    model.back_execstack = build_exec_stack(model.back_strstack, func_dict) # executable function references
    
    model.update_strstack = build_update_string_stack(hp) # string names
    model.update_execstack = build_exec_stack(model.update_strstack, func_dict) # executable function references

    model.cost_function = setup_functions!(hp)

    !hp.quiet && println("Create model completed")
end


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

        if hp.dropout && (hp.droplim[layer] < 1.0)
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

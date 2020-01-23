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

    hp.alphamod = hp.alpha

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


# argfilt methods to pass in the training loop
    # eventually we can replace these with a code generator
function argfilt(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
    bn::Batch_norm_params, hl::Int, fn::typeof(affine!))
    (dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
end
function argfilt(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
    bn::Batch_norm_params, hl::Int, fn::typeof(affine_nobias!))  #TODO: we can take bias out in layer_functions.jl
    (dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
end
function argfilt(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
    bn::Batch_norm_params, hl::Int, fn::typeof(relu!))
    (dat.a[hl], dat.z[hl])
end
function argfilt(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
    bn::Batch_norm_params, hl::Int, fn::typeof(softmax!))
    (dat.a[hl], dat.z[hl])
end
function argfilt(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
    bn::Batch_norm_params, hl::Int, fn::typeof(batch_norm_fwd!))
    (dat, bn, hp, hl)
end
function argfilt(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
    bn::Batch_norm_params, hl::Int, fn::typeof(batch_norm_fwd_predict!))
    (dat, bn, hp, hl)
end

function create_funcs(dat, nnw, bn, hp)
    dat.z::T_model_data = dat.z; 
    dat.a::T_model_data = dat.a; 
    dat.grad::T_model_data = dat.grad
    nnw.theta::T_theta = nnw.theta; 
    nnw.bias::T_bias = nnw.bias;
    dat::Model_data = dat; 
    bn::Batch_norm_params = bn; 
    hp::Hyper_parameters = hp;

    # # curried/closure function definitions for feed forward
    # # affine
    # affine!(hl) = begin  # let 
    #         # dat.z::T_model_data = dat.z; dat.a::T_model_data = dat.a; nnw.theta = nnw.theta; nnw.bias = nnw.bias;
    #         GeneralNN.affine!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
    #     end
    # affine_nobias!(hl) = begin # let
    #         # dat.z::T_model_data = dat.z; dat.a::T_model_data = dat.a; nnw.theta = nnw.theta; nnw.bias = nnw.bias
    #         GeneralNN.affine_nobias!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
    #     end
    # # dropout
    
    # # activation
    # sigmoid!(hl) = begin #let
    #         # dat.a::T_model_data = dat.a; dat.z::T_model_data = dat.z
    #         GeneralNN.sigmoid!(dat.a[hl], dat.z[hl])
    #     end
    # tanh_act!(hl) = begin #let
    #         # dat.a::T_model_data = dat.a; dat.z::T_model_data = dat.z
    #         GeneralNN.tanh_act!(dat.a[hl], dat.z[hl])
    #     end
    # l_relu!(hl) = begin # let
    #         # dat.a::T_model_data = dat.a; dat.z::T_model_data = dat.z
    #         GeneralNN.l_relu!(dat.a[hl], dat.z[hl])
    #     end
    # relu!(hl) = begin #let
    #         # dat.a::T_model_data = dat.a; dat.z::T_model_data = dat.z
    #         GeneralNN.relu!(dat.a[hl], dat.z[hl])
    #     end

    # # classification
    # softmax!(hl) = begin #let
    #         # dat.a::T_model_data = dat.a; dat.z::T_model_data = dat.z
    #         GeneralNN.softmax!(dat.a[hl], dat.z[hl])
    #     end
    # logistic!(hl) = begin #let
    #         # dat.a::T_model_data = dat.a; dat.z::T_model_data = dat.z
    #         GeneralNN.logistic!(dat.a[hl], dat.z[hl])
    #     end
    # regression!(hl) = begin #let
    #         # dat.a::T_model_data = dat.a; dat.z::T_model_data = dat.z
    #         GeneralNN.regression!(dat.a[hl], dat.z[hl])
    #     end

    # # batch norm
    # batch_norm_fwd!(hl) = begin #let
    #         # dat::Model_data = dat; bn::Batch_norm_params = bn; hp::Hyper_parameters = hp;
    #         GeneralNN.batch_norm_fwd!(dat, bn, hp, hl)
    #     end

    # # optimization
    # dropout_fwd!(hl) = begin #let
    #         # dat::Model_data = dat; hp::Hyper_parameters = hp; 
    #         dropout_fwd!(dat, hp, hl)
    #     end

    # # curried function definitions for back propagation
    # # gradient of activation functions
    # affine_gradient(hl) = begin #let
    #         # dat::Model_data = dat
    #         affine_gradient(dat, hl)  # not using yet TODO RENAME FOR CONSISTENCY; can't work as is
    #     end
    # sigmoid_gradient!(hl) = begin #let
    #         # dat.grad::T_model_data = dat.grad; dat.z::T_model_data = dat.z
    #         sigmoid_gradient!(dat.grad[hl], dat.z[hl])
    #     end
    # tanh_act_gradient!(hl) = begin #let
    #         # dat.grad::T_model_data = dat.grad; dat.z::T_model_data = dat.z
    #         l_relu_gradient!(dat.grad[hl], dat.z[hl])
    #     end
    # l_relu_gradient!(hl) = begin #let
    #         # dat.grad::T_model_data = dat.grad; dat.z::T_model_data = dat.z
    #         l_relu_gradient!(dat.grad[hl], dat.z[hl])
    #     end
    # relu_gradient!(hl) = begin #let
    #         # dat.grad::T_model_data = dat.grad; dat.z::T_model_data = dat.z
    #         relu_gradient!(dat.grad[hl], dat.z[hl])
    #     end

    # # optimization
    # dropout_back!(hl) = begin #let
    #         # dat::Model_data = dat
    #         dropout_back!(dat, hl)
    #     end

    # indexable function container  TODO start with just feed fwd
    func_dict = Dict(   # activation
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

                        # gradient
                        "affine_gradient" => affine_gradient!,
                        "sigmoid_gradient" => sigmoid_gradient!,
                        "tanh_act_gradient" => l_relu_gradient!,
                        "l_relu_gradient" => l_relu_gradient!,
                        "relu_gradient" => relu_gradient!,

                        # optimization
                        "dropout_back" => dropout_back!
                    )
end


function mini_eval(dat)
    do_relu = (GeneralNN.relu!, Meta.parse("(dat.a[hl], dat.z[hl])"))
end


function build_ff_string_stack!(model, hp, dat)
    strstack = []
    n_hid = length(hp.hidden)
    n_layers = n_hid + 2

    #input layer
    lr = 1
        push!(strstack, String[]) # layer_group for input layer
        if hp.dropout && (hp.droplim[1] < 1.0)
            push!(strstack[lr], "dropout")
        end

    # hidden layers
    for lr = 2:n_hid+1
        layer_group = String[]

        if hp.dropout && (hp.droplim[1] < 1.0)
            push!(layer_group, "dropout")
        end

        # affine either or...
        hp.do_batch_norm && push!(layer_group, "affine_nobias") # true
        hp.do_batch_norm || push!(layer_group, "affine") # false
        # batch_norm_fwd 
        hp.do_batch_norm && push!(layer_group, "batch_norm_fwd")
        # unit_function or activation --> updates in place
        unit_function = 
            if hp.hidden[lr-1][1] == "sigmoid"
                "sigmoid"
            elseif hp.hidden[lr-1][1] == "l_relu"
                "l_relu"
            elseif hp.hidden[lr-1][1] == "relu"
                "relu"
            elseif hp.hidden[lr-1][1] == "tanh"
                "tanh_act"
            end
        push!(layer_group, unit_function)
        # dropout --> updates in place
        if hp.dropout && (hp.droplim[hl] < 1.0)
            push!(strstack[i], "dropout")
        end        

        # done with layer_group for current hidden layer
        push!(strstack, layer_group) 
    end

    #   output layer

    i = n_hid + 2
        layer_group = String[]

        push!(layer_group, "affine")
        # classify_function --> updates in place
        classify_function = 
            if dat.out_k > 1  # more than one output (unit)
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

        model.ff_strstack = strstack
end   


function build_ff_exec_stack!(model, func_dict)
    execstack = []
    strstack = model.ff_strstack
    for i in 1:size(strstack,1)
        push!(execstack,[])
        for r in strstack[i]
            push!(execstack[i], func_dict[r])
        end
    end
    model.ff_execstack = execstack
end


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

    # TODO DO WE NEED THIS? allow different functions at each appropriate layer
    unit_function! = Array{Function}(undef, n_layers)
    gradient_function! = Array{Function}(undef, n_layers)
    reg_function! = Array{Function}(undef, n_layers)
    dropout_fwd_function! = Array{Function}(undef, n_layers)
    dropout_back_function! = Array{Function}(undef, n_layers)

    for layer in 2:n_layers-1 # for hidden layers: layers 2 through output - 1
        hl = hidden_layer = layer - 1
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

    # for layer = 1:n_layers-1 # input layer and hidden layers
    #     dropout_fwd_function![layer] = 
    #         if hp.dropout && (hp.droplim[hl] < 1.0)
    #             dropout_fwd!
    #         else
    #             noop
    #         end
    # end


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

    # classify_function! = 
    #     if dat.out_k > 1  # more than one output (unit)
    #         if hp.classify == "sigmoid"
    #             sigmoid!
    #         elseif hp.classify == "softmax"
    #             softmax!
    #         else
    #             error("Function to classify multiple output labels must be \"sigmoid\" or \"softmax\".")
    #         end
    #     else
    #         if hp.classify == "sigmoid" || hp.classify == "logistic"
    #             logistic!  # for one output label
    #         elseif hp.classify == "regression"
    #             regression!
    #         else
    #             error("Function to classify single output must be \"sigmoid\", \"logistic\" or \"regression\".")
    #         end
    #     end

    cost_function = 
        if hp.classify=="regression" 
            mse_cost 
        elseif hp.classify == "softmax"
            softmax_cost
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
    return function batch_norm_fwd_curry(dat, hl)  # arguments that will be passed in when new function called
                batch_norm_fwd!(dat, bn, hp, hl)
           end  # actual name of the resulting function set to return result of create_curry function
end


function create_curry_batch_norm_fwd_predict!(hp, bn)  # arguments that are captured and "built-in" to curried function
    return function create_curry_batch_norm_fwd_predict_curry(dat, hl)  # arguments that will be passed in when new function called
                batch_norm_fwd_predict!(dat, bn, hp, hl)
           end  # actual name of the resulting function set to return result of create_curry function
end


function create_curry_batch_norm_back!(hp, bn, nnw)  # arguments that are captured and "built-in" to curried function
    return function create_curry_batch_norm_back_curry(dat, hl)  # arguments that will be passed in when new function called
                batch_norm_back!(nnw, dat, bn, hl, hp)
           end  # actual name of the resulting function set to return result of create_curry function
end





###################################################################
#   experimental
###################################################################

feedfwd_funcs = Dict(
    "relu" => (f=relu!, args=()),
    )

backprop_funcs = Dict(

    )


function build_model()
    model = nothing # TODO
    return model
end
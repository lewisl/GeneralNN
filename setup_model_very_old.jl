using JLD2
using Printf
using LinearAlgebra


function build_model()
    model = nothing # TODO
    return model
end

####################################################################
#  verify hyper_parameter inputs from TOML or function arguments
####################################################################

# Works with simple toml file containing all arguments at top level
function args_verify(argsdict)
    required = [:epochs, :hidden]
    all(i -> i in Symbol.(keys(argsdict)), required) || error("Missing a required argument: epochs or hidden")

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


function args_verify_tables(argsdict)
    # tables are: layer, training, results, plotting

    #required tables
    required_tables = [:layer, :training]
    all(i -> i in Symbol.(keys(argsdict)), required_tables) || error("Missing a required table: layer, training, or output.")

    args_training = argsdict["training"]
    args_layer = argsdict["layer"]

    args_verify_training(args_training)
    args_verify_layer(args_layer)
    
    # optional tables (replaced with defaults if not present)
    args_results = get(argsdict, "results", nothing)
    args_plotting = get(argsdict, "plotting", nothing)

    args_verify_results()
    args_verify_plotting()

end


function args_verify_layer(argsdict)
    # layer.output
    key_output = pop!(argsdict,"output",nothing)
    key_output === nothing && error("Missing input for layer.output")
    # check for keys and values of layer.output_layer
    try
        if key_output["classify"] in ["softmax", "sigmoid", "logistic", "regression"]
        else
            error("classify must be one of softmax, sigmoid, logistic or regression.")
        end
    catch
        error("Missing argument classify for output layer.")
    end

    # hidden layers
        try
            hls = parse.(Int, collect(keys(argsdict)))
        catch
            error("One or more hidden layers is not an integer.")
        end
        hls = sort(hls)
        println(hls)
        hls[1]:hls[end] != 1:size(hls,1) && error("Hidden layer numbers are not a sequence from 1 to $(size(hls,1)).")
    # check for keys and and values of hidden layers
    for (lyr,lyrdict) in argsdict
        for (k,v) in lyrdict
            checklist = get(valid_layers, Symbol(k), nothing)
            checklist === nothing && error("Parameter name is not valid: $k")

            for tst in checklist
                # eval the tst against the value in argsdict
                result = all(tst.f(v, tst.check)) 
                warn = get(tst, :warn, false)
                msg = get(tst, :msg, warn ? "Input argument not ideal: $k: $v" : "Input argument not valid: $k: $v" )
                !result && (warn ? @warn(msg) : error("Layer $lyr", " ", msg))
            end
        end
    end       
      
    # put the dict back together after popping
    argsdict["output"] = key_output
end



function args_verify_training(argsdict)
    required = [:epochs, :hidden]
    all(i -> i in Symbol.(keys(argsdict)), required) || error("Missing a required argument: epochs or hidden")

    for (k,v) in argsdict
        checklist = get(valid_training, Symbol(k), nothing)
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

# functions used to verify input values in result = all(test.f(v, tst.check))
    eqtype(item, check) = typeof(item) == check
    ininterval(item, check) = check[1] .<= item .<= check[2] 
    ininterval2(item, check) = check[1] .<= map(x -> parse(Int, x[2]), item) .<= check[2] # 2nd element of tuple item
    oneof(item::String, check) = lowercase(item) in check 
    oneof(item::Real, check) = item in check
    lengthle(item, check) = length(item) <= check 
    lengtheq(item, check) = length(item) == check
    lrndecay(item, check) = ininterval(item[1],check[1]) && ininterval(item[2], check[2])

# for stats
valid_stats = ["learning", "cost", "train", "test", "batch", "epoch"]
function checkstats(item, _)
    if length(item) > 1
        ok1 = all(i -> i in valid_stats, lowercase.(item)) 
        ok2 = allunique(item) 
        ok3 = !("batch" in item && "epoch" in item)  # can't have both, ok to have neither
        ok = all([ok1, ok2, ok3])
    elseif length(item) == 1
        ok = item[1] in ["", "none"] 
    else
        ok = true
    end
    return ok
end

# for training
    # key for each input param; value is list of checks as tuples 
    #     check keys are f=function, check=values, warn can be true--default is false, msg="something"
    #     example:  :alpha =>  [(f=eqtype, check=Float64), (f=ininterval, check=(.000001, 9.0), warn=true)]
const valid_training = Dict(
          :epochs => [(f=eqtype, check=Int), (f=ininterval, check=(1,9999))],
          :alpha =>  [(f=eqtype, check=Float64), (f=ininterval, check=(.000001, 9.0), warn=true)],
          :reg =>  [(f=oneof, check=["l2", "l1", "maxnorm", "", "none"])],
          :maxnorm_lim =>  [(f=eqtype, check=Array{Float64, 1})],
          :lambda =>  [(f=eqtype, check=Float64), (f=ininterval, check=(0.0, 5.0))],
          :learn_decay =>  [(f=eqtype, check=Array{Float64, 1}), (f=lengtheq, check=2), 
                            (f=lrndecay, check=((.1,1.0), (1.0,20.0)))],
          :mb_size_in =>  [(f=eqtype, check=Int), (f=ininterval, check=(0,1000))],
          :norm_mode => [(f=oneof, check=["standard", "minmax", "", "none"])] ,
          :dobatch =>  [(f=eqtype, check=Bool)],
          :do_batch_norm =>  [(f=eqtype, check=Bool)],
          :opt =>  [(f=oneof, check=["momentum", "rmsprop", "adam", "", "none"])],
          :opt_params =>  [(f=eqtype, check=Array{Float64,1}), (f=ininterval, check=(0.5,1.0))],
          :dropout =>  [(f=eqtype, check=Bool)],
          :droplim =>  [(f=eqtype, check=Array{Float64, 1}), (f=ininterval, check=(0.2,1.0))],
          :stats =>  [(f=checkstats, check=nothing)],
          :plot_now =>  [(f=eqtype, check=Bool)],
          :quiet =>  [(f=eqtype, check=Bool)],
          :initializer => [(f=oneof, check=["xavier", "uniform", "normal", "zero"], warn=true, 
                            msg="Setting to default: xavier")] ,
          :scale_init =>  [(f=eqtype, check=Float64)],
          :bias_initializer => [(f=eqtype, check=Float64, warn=true, msg="Setting to default 0.0"),
                                (f=ininterval, check=(0.0,1.0), warn=true, msg="Setting to default 0.0")],
          :sparse =>  [(f=eqtype, check=Bool)]
        )

# for layers
const valid_layers = Dict(
          :activation =>  [(f=oneof, check=["sigmoid", "l_relu", "relu", "tanh"]), 
                           (f=oneof, check=["l_relu", "relu"], warn=true, 
                            msg="Better results obtained with relu using input and/or batch normalization. Proceeding...")],
          :units => [(f=eqtype, check=Int), (f=ininterval, check=(1, 8192))],
          :linear => [(f=eqtype, check=Bool)]
        )

# for simple file containing all arguments at top level
    # key for each input param; value is list of checks as tuples 
    #     check keys are f=function, check=values, warn can be true--default is false, msg="something"
    #     example:  :alpha =>  [(f=eqtype, check=Float64), (f=ininterval, check=(.000001, 9.0), warn=true)]
const valid_toml = Dict(
          :epochs => [(f=eqtype, check=Int), (f=ininterval, check=(1,9999))],
          :hidden =>  [ (f=lengthle, check=11), (f=eqtype, check=Array{Array, 1}),
                       (f=ininterval2, check=(1,8192))],                     
          :alpha =>  [(f=eqtype, check=Float64), (f=ininterval, check=(.000001, 9.0), warn=true)],
          :reg =>  [(f=oneof, check=["l2", "l1", "maxnorm", "", "none"])],
          :maxnorm_lim =>  [(f=eqtype, check=Array{Float64, 1})],
          :lambda =>  [(f=eqtype, check=Float64), (f=ininterval, check=(0.0, 5.0))],
          :learn_decay =>  [(f=eqtype, check=Array{Float64, 1}), (f=lengtheq, check=2), 
                            (f=lrndecay, check=((.1,1.0), (1.0,20.0)))],
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
          :stats =>  [(f=checkstats, check=nothing)],
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
        if k == "hidden"  # special case because of TOML limitation:  change [["relu", "80"]] to [("relu", 80)]
            setproperty!(hp, Symbol(k), map( x -> (x[1], parse(Int, x[2])), v) ) 
        else
            setproperty!(hp, Symbol(k), v)
        end
    end

    return hp
end


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
    bn.delta_v_gam = [zeros(i) for i in k]
    bn.delta_s_gam = [zeros(i) for i in k]
    bn.delta_v_bet = [zeros(i) for i in k]
    bn.delta_s_bet = [zeros(i) for i in k]
    bn.mu = [zeros(i) for i in k]  # same size as bias = no. of layer units
    bn.mu_run = [zeros(i) for i in k]
    bn.stddev = [zeros(i) for i in k]
    bn.std_run = [zeros(i) for i in k]

end


feedfwd_funcs = Dict(
    "relu" => (f=relu!, args=()),
    )

backprop_funcs = Dict(

    )
using JLD2
using Printf
using LinearAlgebra



"""
    function setup_params(epochs::Int64, hidden::Array{Tuple{String,Int64},1}; alpha::Float64=0.35
    mb_size_in::Int64=0, lambda::Float64=0.01, classify::String="softmax", norm_mode::String="none"
    opt::String="", opt_params::Array{Float64,1}=[0.9,0.999], do_batch_norm::Bool=false
    reg::String="L2", dropout::Bool=false, droplim::Array{Float64,1}=[0.5], maxnorm_lim,
    plots::Array{String,1}=["Training", "Learning"], learn_decay::Array{Float64,1}=[1.0, 1.0])

    Note that there is a convenience method that reads all of the inputs from a .TOML file. 
    This is MUCH to be preferred:

        function setup_params(argsfile::String)
        This method gets input arguments from a TOML file. This method does no
        error checking except, optionally, for valid argnames or missing required args.    

    returns:           hp, a Hyper_parameters struct of all of the hyper_parameters. 

    required inputs:
        epochs          ::= Int64, no. of training passes to perform
        hidden          ::= array of tuple(Sting,Int) containing type and number of units in each hidden layer;
                            make sure to use an array even with 1 hidden layer as in [("relu",40)];
                            use [0] to indicate no hidden layer (typically for linear regression)

    optional named parameters:
        alpha           ::= learning rate
        lambda          ::= regularization rate
        dobatch         ::= Bool, true to use minibatch training
        mb_size_in      ::= minibatch size=>integer
        norm_mode       ::= "standard", "minmax" or "" => normalize inputs
        do_batch_norm   ::= true or false => normalize each hidden layers linear outputs
        opt             ::= one of "Momentum", "RMSProp", "Adam" or "".  default is blank string "".
        opt_output      ::= true or false: apply optimization to the weights of the output layer
        opt_batch_norm  ::= Bool.  Apply optimization to batchnorm_params (true) or not (false).
        opt_params      ::= parameters used by Momentum, RMSProp, or Adam
                           Momentum, RMSProp: one floating point value as [.9] (showing default)
                           Adam: 2 floating point values as [.9, .999] (showing defaults)
                           Note that epsilon is ALWAYS set to 1e-8
                           To accept defaults, don't input this parameter or use []
        classify        ::= "softmax", "sigmoid", "logistic", or "regression" for only the output layer

        reg             ::= type of regularization, must be one of "L1", "L2", "Maxnorm", ""
        maxnorm_lim     ::= Array{Float64,1}, array of limits set for hidden layers + output layer
        dropout         ::= true to use dropout network or false
        droplim         ::= array of values between 0.5 and 1.0 determines how much dropout for
                            hidden layers and input layer (ex: [0.8] or [0.8,0.9, 1.0]).  A single
                            value will be applied to all hidden layers.  If fewer values than layers, then the
                            last value extends to remaining layers.
        learn_decay     ::= array of 2 float values:  first is > 0.0 and <= 1.0 which is factor to reduce 
                            learning rate; second is >= 1.0 and <= 10.0 for number of times to 
                            reduce learning rate (alpha).  Ex: [.5, 2.0] reduces alpha in 1/2 after 1/2 of the
                            epochs.
                            [1.0, 1.0] signals don't do learning decay
        sparse          ::Bool. If true, input data will be treated and maintained as SparseArrays.
        initializer     ::= "xavier", "uniform", "normal" or "zero" used to set how Wgts, not including bias, are initialized.
        scale_init      ::= Float64. Scale the initialization values for the parameters.
        bias_initializer::= Float64. 0.0, 1.0 or float inbetween. Initial value for bias terms.
        stats           ::= determines training statistics collected and plotted
                            any choice of ["learning", "cost", "train", "test"];
                            for no stats use [""] or ["none"]
                            include "batch" to collect stats on each minibatch and/or "epoch" (the default) for stats per epoch
                            WARNING:  stats for every batch is SLOW.
        plot_now        ::Bool.  If true, plot training stats immediately and save the stats that contains stats 
                            gathered while running the training.  You can plot the file separately, later.
        quiet           ::Bool. true is default to suppress progress messages during preparation and training.

        The last 3 items are not training hyperparameters, but control plotting and the process.


    The TOML file need only contain argument values that you desire to set
    differently than the defaults. 

    Here is an example of a correct TOML file containing every permitted argument:

        alpha =   0.74
        lambda =  0.000191
        hidden = [("relu",100)]                  # up to 9 hidden layers, up to 4096 units per layer
        reg =  "L2"                    # or "L1", "Maxnorm"
        maxnorm_lim = []
        classify = "softmax"           # or "sigmoid", "regression", "logistic"
        dropout = false
        droplim = [1.0,0.8,1.0]
        epochs =  24
        mb_size_in =   50
        norm_mode =   "none"           # or "standard", "minmax"
        opt =   "adam"                 # or "momentum", "RMSProp"
        opt_batch_norm = false
        opt_params = [0.9, 0.999]  
        learn_decay = [0.5,4.0]
        dobatch = true
        do_batch_norm =  true
        reshuffle = true
        sparse = false
        initializer = "xavier"          # or "normal", "uniform"
        scale_init = 2.0, 
        bias_initializer  = 0.0
        quiet = true
        stats = ["Train", "Learning", "Test", "epoch", "batch"]           
        plot_now = true                    

"""
function setup_params(       
                epochs::Int64, 
                hidden::Array{Tuple{String,Int64},1}=[("none",0)];   # required
                alpha::Float64=0.35,                              # named, optional if defaults ok
                mb_size_in::Int64=0, 
                lambda::Float64=0.01, 
                classify::String="softmax", 
                norm_mode::String="none", 
                opt::String="", 
                opt_output::Bool=false,
                opt_batch_norm=false,
                opt_params::Array{Float64,1}=[0.9,0.999], 
                dobatch=false, 
                do_batch_norm::Bool=false, 
                reshuffle::Bool=false,
                reg::String="", 
                maxnorm_lim::Array{Float64,1}=Float64[], 
                dropout::Bool=false, 
                droplim::Array{Float64,1}=[], 
                stats::Array{String,1}=["Training", "Learning"], 
                learn_decay::Array{Float64,1}=[1.0, 1.0], 
                plot_now::Bool=false, 
                sparse::Bool=false, 
                initializer::String="xavier", 
                scale_init::Float64=2.0, 
                bias_initializer::Float64=0.0, 
                quiet=true
            )

    # this method serves to validate the input parameters and populate struct hp
    # for convenience use the TOML file input method

    # create dict from input args
    argsdict = Dict(
        "epochs"            => epochs,
        "hidden"            => hidden,
        "alpha"             => alpha,        
        "mb_size_in"        => mb_size_in, 
        "lambda"            => lambda, 
        "classify"          => classify, 
        "norm_mode"         => norm_mode, 
        "opt"               => opt, 
        "opt_output"        => opt_output,
        "opt_batch_norm"    => opt_batch_norm,
        "opt_params"        => opt_params, 
        "dobatch"           => dobatch, 
        "do_batch_norm"     => do_batch_norm, 
        "reshuffle"         => reshuffle,
        "reg"               => reg, 
        "maxnorm_lim"       => maxnorm_lim, 
        "dropout"           => dropout, 
        "droplim"           => droplim, 
        "stats"             => stats, 
        "learn_decay"       => learn_decay, 
        "plot_now"          => plot_now, 
        "sparse"            => sparse, 
        "initializer"       => initializer, 
        "scale_init"        => scale_init, 
        "bias_initializer"  => bias_initializer, 
        "quiet"             => quiet  
        )

    args_verify(argsdict)  # no return; errors out at first error

    hp = build_hyper_parameters(argsdict)

    # validate_datafiles(datalist) # no return; errors out if errors
 
    return hp
end


function setup_params(argsfile::String)
################################################################################
#   This method gets input arguments from a TOML file. This method does no
#   error checking except, optionally, for valid argnames or missing required args. 
################################################################################

    if splitext(argsfile)[end] == ".toml"
        argsdict = TOML.parsefile(argsfile)
    else
        error("File extension must be .toml")
    end

    args_verify(argsdict)  # no return; errors out at first error

    hp = build_hyper_parameters(argsdict)

    return hp
end


####################################################################
#  verify hyper_parameter inputs from TOML or function arguments
####################################################################

# Works with dict containing all arguments at top level
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

    function ininterval2(item, check)
        map(x -> check[1] < x[2] < check[2], item)
    end

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
          :reshuffle => [(f=eqtype, check=Bool)],
          :opt =>  [(f=oneof, check=["momentum", "rmsprop", "adam", "", "none"])],
          :opt_output => [(f=eqtype, check=Bool)],
          :opt_batch_norm => [(f=eqtype, check=Bool)],
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
          :hidden =>  [ (f=lengthle, check=11), (f=eqtype, check=Vector{Vector{Union{Int64, String}}}),
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
          :reshuffle => [(f=eqtype, check=Bool)],
          :opt =>  [(f=oneof, check=["momentum", "rmsprop", "adam", "", "none"])],
          :opt_output => [(f=eqtype, check=Bool)],
          :opt_batch_norm => [(f=eqtype, check=Bool)],
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
            setproperty!(hp, Symbol(k), map(x -> Tuple(x), v)) 
        else
            setproperty!(hp, Symbol(k), v)
        end
    end

    hp.n_layers = length(hp.hidden) + 2

    return hp
end


"""
    function build_argsdict(hp::Hyper_parameters)

Input: a Hyper_parameters object, which is a mutable struct

Returns: a Dict of all of the hyperparameter fields

"""
function build_argsdict(hp::Hyper_parameters)
    ret = Dict()
    for hpsym in fieldnames(typeof(hp))
        hpitem = String(hpsym)
        ret[hpitem] = getfield(hp,hpsym)
    end
    return ret
end


"""
    function prep_training(hp, n)
    
Collect parameter setup for batch_size, learn_decay, dropout, 
and Maxnorm regularization in one function.
"""
function prep_training!(hp, n)
    !hp.quiet && println("prep training beginning")
    !hp.quiet && println("hp.dobatch: ", hp.dobatch)

    setup_batch_size!(hp, n)
    setup_learn_decay!(hp)
    hp.dropout && (setup_dropout!(hp))
    (titlecase(hp.reg) == "Maxnorm") && (setup_maxnorm!(hp))

    !hp.quiet && println("end of setup_model: hp.dobatch: ", hp.dobatch)
end


function setup_batch_size!(hp, n)
    if hp.dobatch
        @info("Be sure to shuffle training data when using minibatches.\n  Use utility function shuffle_data! or your own.")
        if hp.mb_size_in < 1
            hp.mb_size_in = n  
            hp.dobatch = false    # user provided incompatible inputs
        elseif hp.mb_size_in >= n
            @warn("Wrong size for minibatch training. Proceeding with full batch training.")
            hp.mb_size_in = n
            hp.dobatch = false   # user provided incompatible inputs
        end 
        hp.mb_size = hp.mb_size_in  # start value for hp.mb_size; changes if last minibatch is smaller
        hp.do_batch_norm = hp.dobatch ? hp.do_batch_norm : false  
    else
        hp.mb_size_in = n
        hp.mb_size = float(n)
    end
end


function setup_learn_decay!(hp)
    # requires error checking in setup_params.jl
    if hp.learn_decay == [1.0, 1.0]
        hp.do_learn_decay = false
    elseif hp.learn_decay == []
        hp.do_learn_decay = false
    else
        hp.do_learn_decay = true  
        hp.learn_decay = [hp.learn_decay[1], floor(hp.learn_decay[2])]
    end
end


function setup_dropout!(hp)
    # dropout parameters: droplim is in hp (Hyper_parameters),
    #    dropout_random and dropout_mask are in mb or train (Model_data)
    # set a droplim for each layer 
    if length(hp.droplim) == length(hp.hidden) + 2  # droplim for every layer
        if hp.droplim[end] != 1.0
            @warn("Poor performance with dropout on output layer, continuing without.")
            hp.droplim[end] = 1.0
        end
        if hp.droplim[1] != 1.0
            @warn("Inconsistent performance with dropout on input layer, continuing.")
        end       
    elseif length(hp.droplim) == length(hp.hidden) + 1 # droplim for input and hidden layers
        if hp.droplim[1] != 1.0
            @warn("Inconsistent performance with dropout on input layer, continuing.")
        end       
        hp.droplim = [hp.droplim..., 1.0]  # keep all units in output layer
    elseif length(hp.droplim) < length(hp.hidden)  # pad droplim for all hidden layers
        for i = 1:length(hp.hidden)-length(hp.droplim)
            push!(hp.droplim,hp.droplim[end]) 
        end
        hp.droplim = [1.0, hp.droplim..., 1.0] # use all units for input and output layers
    else
        @warn("More drop limits provided than total network layers, use dropout ONLY for hidden layers.")
        hp.droplim = hp.droplim[1:length(hp.hidden)]  # truncate
        hp.droplim = [1.0, hp.droplim..., 1.0] # placeholders for input and output layers
    end
end


function setup_maxnorm!(hp)
    hp.reg = "Maxnorm"
    if isempty(hp.maxnorm_lim)
        @warn("Values in Float64 array must be set for maxnormlim to use Maxnorm Reg, continuing with no regularizaiton.")
        hp.reg = ""
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

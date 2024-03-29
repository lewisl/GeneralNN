# some useful shorthands for complex array types
const T_model_data = Array{Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}
const T_theta = Array{Array{Float64,2},1}
const T_bias = Array{Array{Float64,1},1}
const T_union_dense_sparse_array = Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}}
const T_array_subarray = Union{Array{Float64}, SubArray{Float64}}



"""
struct Wgts holds model parameters learned by training and model metadata
"""
mutable struct Wgts              # we will use nnw as the struct variable
    theta::Array{Array{Float64,2},1}
    bias::Array{Array{Float64,1},1}
    delta_th::Array{Array{Float64,2},1}
    delta_b::Array{Array{Float64,1},1}
    # optimization weighted average of gradient: momentum, rmsprop, Adam
    delta_v_th::Array{Array{Float64,2},1}  
    delta_v_b::Array{Array{Float64,1},1}  
    delta_s_th::Array{Array{Float64,2},1}  
    delta_s_b::Array{Array{Float64,1},1}  
    theta_dims::Array{Tuple{Int64, Int64},1}
    output_layer::Int64
    ks::Array{Int64,1}                     # number of output units in each layer (e.g., features for input layer)
                                           #      = the no. of rows in the weight matrix for each layer
    norm_factors::Tuple{Array{Float64,2},Array{Float64,2}}   # note: each array is 1 row by 2 cols
    # calculate dropout mask for training
    dropout_mask::Array{Array{Bool,1}, 1}       # boolean filter for dropout--dims of a


    Wgts() = new(               # empty constructor
        Array{Array{Float64,2},1}(undef, 0),    # theta::Array{Array{Float64,2}}
        Array{Array{Float64,2},1}(undef, 0),    # bias::Array{Array{Float64,1}}
        Array{Array{Float64,2},1}(undef, 0),    # delta_th
        Array{Array{Float64,1},1}(undef, 0),    # delta_b
        Array{Array{Float64,2},1}(undef, 0),    # delta_v_th
        Array{Array{Float64,1},1}(undef, 0),    # delta_v_b
        Array{Array{Float64,2},1}(undef, 0),    # delta_s_th
        Array{Array{Float64,1},1}(undef, 0),    # delta_s_b
        Array{Tuple{Int, Int},1}(undef, 0),     # theta_dims::Array{Array{Int64,2}}
        3,                                      # output_layer
        Array{Int64,1}(undef, 0),               # k
        ([0.0 0.0], [1.0 0.0]),                 # norm_factors (mean, std)
        Array{Array{Bool,1},1}(undef, 0)        # dropout_mask   Array{Array{Bool,2},1}

    )
end


"""
struct Hyper_parameters holds hyper_parameters used to control training
"""
mutable struct Hyper_parameters          # we will use hp as the struct variable
    alpha::Float64              # learning rate
    alphamod::Float64           # optionally adjust learning rate with learn_decay
    lambda::Float64             # L2 regularization rate
    hidden::Array{Tuple{String,Int64},1}       # array of ("unit", number) for hidden layers
    n_layers::Int64
    b1::Float64                 # 1st optimization for momentum or Adam
    b2::Float64                 # 2nd optimization parameter for Adam
    ltl_eps::Float64            # use in denominator with division of very small values to prevent overflow
    dobatch::Bool               # simple flag on whether to do minibatch training
    do_batch_norm::Bool         # true or false
    reshuffle::Bool
    norm_mode::String           # "", "none", "standard", or "minmax"
    dropout::Bool               # true or false to choose dropout network
    droplim::Array{Float64,1}   # the probability a node output is kept
    reg::String                 # L1, L2, maxnorm, or "none"
    maxnorm_lim::Array{Float64,1}# [] with limits for hidden layers and output layer
    opt::String                 # Adam or momentum or "none" or "" for optimization
    opt_output::Bool               # appy optimization to output layer
    opt_batch_norm::Bool        # don't optimize batchnorm params if optimizing training weights
    opt_params::Array{Float64,1}# parameters for optimization
    classify::String            # behavior of output layer: "softmax", "sigmoid", or "regression"
    mb_size::Int64              # minibatch size--calculated; last mini-batch may be smaller
    mb_size_in::Int64           # input of requested minibatch size:  last actual size may be smaller
    epochs::Int64               # number of "outer" loops of training
    do_learn_decay::Bool        # step down the learning rate across epochs
    learn_decay::Array{Float64,1}  # reduction factor (fraction) and number of steps
    sparse::Bool
    initializer::String         # "xavier" or "zero"
    scale_init::Float64         # varies with initializer method: 2.0 for xavier, around .15 for others
    bias_initializer::Float64   # 0.0, 1.0, between them
    quiet::Bool                 # display progress messages or not
    stats::Array{String, 1}     # not a hyper_parameter, choice of stats data to collect during training
    plot_now::Bool


    Hyper_parameters() = new(       # constructor with defaults--we use hp as the struct variable
        0.35,           # alpha -- OK for nn. way too high for linear regression
        0.35,           # alphamod
        0.01,           # lambda
        [("none",0)],   # hidden
        0,              # n_layers
        0.9,            # b1
        0.999,          # b2
        1e-8,           # ltl_eps
        false,          # dobatch
        false,          # do_batch_norm
        false,          # reshuffle
        "none",         # norm_mode
        false,          # dropout
        [],             # droplim
        "",             # reg
        Float64[],      # maxnorm_lim
        "",             # opt
        false,           # opt_output
        false,          # opt_batch_norm
        [],             # opt_params
        "sigmoid",      # classify
        0,              # mb_size
        50,             # mb_size_in  
        1,              # epochs
        false,          # do_learn_decay
        [1.0, 1.0],     # learn_decay
        false,          # sparse
        "xavier",       # initializer
        2.0,            # scale_init
        0.0,            # bias_initializer
        true,           # quiet
        ["None"],       # stats
        false          # plot_now

    )
end


"""
Struct Model_data hold examples and all layer outputs-->
pre-allocate to reduce memory allocations and improve speed.
Most of these are 1 dimensional arrays (an element for each layer) of arrays
(the array data values at a layer).
"""
mutable struct Model_data               # we will use train for inputs and test for test data
    # read from training, test, or production data
    inputs::Union{AbstractArray{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}} #::   # in_k features by n examples
    targets::Union{AbstractArray{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}} #::  # labels for each example
    # calculated in feedforward pass
    a::T_model_data  # 
    z::T_model_data
    # calculated in backprop pass
    grad::T_model_data
    epsilon::T_model_data       # dims of a
    # calculcated for batch_norm
    z_norm::T_model_data   # same size as z--for batch_norm
    # descriptive
    n::Int64                                  # number of examples
    in_k::Int64                               # number of input features
    out_k::Int64                              # number of output features
    

    Model_data() = new(                 # empty constructor
        zeros(0,0),          # inputs
        zeros(0,0),          # targets
        [zeros(0,0)],       # a
        [zeros(0,0)],       # z
        [zeros(0,0)],       # grad
        [zeros(0,0)],       # epsilon
        [zeros(0,0)],       # z_norm -- only pre-allocate if batch_norm
        0,                              # n
        0,                              # in_k
        0                               # out_k
    )
end


"""
Struct Batch_view holds views on all model data that will be broken into minibatches
"""
mutable struct Batch_view               # we will use mb for as the variable for minibatches
    # array of views
    a::Array{SubArray{}}  #::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    targets::SubArray{}  #::SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true}
    z::Array{SubArray{}}  #::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    z_norm::Array{SubArray{}}  #::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    grad::Array{SubArray{}}  #::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    epsilon::Array{SubArray{}}  #::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}

    Batch_view() = new(                      # empty constructor
        Array{SubArray{}}[],  # a
        view([0.0],1:1),      # targets
        Array{SubArray{}}[],  # z
        Array{SubArray{}}[],  # z_norm
        Array{SubArray{}}[],  # grad
        Array{SubArray{}}[],  # epsilon
        )
   
end


"""
struct Batch_norm_params holds batch normalization parameters for
feedfwd calculations and backprop training.
"""
mutable struct Batch_norm_params               # we will use bn as the struct variable
    # learned batch parameters to center and scale data
    gam::Array{Array{Float64,1},1}   # scaling parameter for z_norm
    bet::Array{Array{Float64,1},1}   # shifting parameter for z_norm (equivalent to bias)
    delta_gam::Array{Array{Float64,1},1}
    delta_bet::Array{Array{Float64,1},1}
    # for optimization updates of bn parameters
    delta_v_gam::Array{Array{Float64,1},1}  
    delta_s_gam::Array{Array{Float64,1},1}  
    delta_v_bet::Array{Array{Float64,1},1}  
    delta_s_bet::Array{Array{Float64,1},1}  
    # for standardizing batch values
    mu::Array{Array{Float64,1},1}              # mean of z; same size as bias = no. of input layer units
    stddev::Array{Array{Float64,1},1}          # std dev of z;   ditto
    mu_run::Array{Array{Float64,1},1}          # running average of mu
    std_run::Array{Array{Float64,1},1}         # running average of stddev

    Batch_norm_params() = new(           # empty constructor
        Array{Array{Float64,1},1}(undef, 0),    # gam::Array{Array{Float64,1}}
        Array{Array{Float64,1},1}(undef, 0),    # bet::Array{Array{Float64,1}}
        Array{Array{Float64,2},1}(undef, 0),    # delta_gam
        Array{Array{Float64,2},1}(undef, 0),    # delta_bet
        Array{Array{Float64,2},1}(undef, 0),    # delta_v_gam
        Array{Array{Float64,2},1}(undef, 0),    # delta_s_gam
        Array{Array{Float64,2},1}(undef, 0),    # delta_v_bet
        Array{Array{Float64,2},1}(undef, 0),    # delta_s_bet
        Array{Array{Float64,1},1}(undef, 0),    # mu
        Array{Array{Float64,1},1}(undef, 0),    # stddev
        Array{Array{Float64,1},1}(undef, 0),    # mu_run
        Array{Array{Float64,1},1}(undef, 0)     # std_run
    )
end


"""
struct Model_def holds the functions that will run in a 
model based on the hyper_parameters and data
"""
mutable struct Model_def
    ff_strstack::Array{Array{String,1},1}
    ff_execstack::Array{Array{Function,1},1}
    back_strstack::Array{Array{String,1},1}
    back_execstack::Array{Array{Function,1},1}
    update_strstack::Array{Array{String,1},1}
    update_execstack::Array{Array{Function,1},1}
    cost_function::Function


    Model_def() = new(
        [String[]],         # ff_strstack
        [Function[]],       # ff_execstack
        [String[]],         # back_strstack
        [Function[]],       # back_execstack
        [String[]],         # update_strstack
        [Function[]],       # update_execstack
        noop,                # cost_function

    )
end
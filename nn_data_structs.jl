
"""
struct NN_weights holds model parameters learned by training and model metadata
"""
mutable struct NN_weights              # we will use nnp as the struct variable
    theta::Array{Array{Float64,2},1}
    bias::Array{Array{Float64,1},1}
    delta_w::Array{Array{Float64,2},1}
    delta_b::Array{Array{Float64,1},1}
    delta_v_w::Array{Array{Float64,2},1}  # momentum weighted average of gradient--also for Adam
    delta_v_b::Array{Array{Float64,1},1}  # hold momentum weighted average of gradient--also for Adam
    delta_s_w::Array{Array{Float64,2},1}  # s update term for ADAM
    delta_s_b::Array{Array{Float64,1},1}  # s update term for ADAM
    theta_dims::Array{Tuple{Int64, Int64},1}
    output_layer::Int64
    layer_units::Array{Int64,1}
    norm_factors::Tuple{Any, Any}

    NN_weights() = new(               # empty constructor
        Array{Array{Float64,2},1}(undef, 0),    # theta::Array{Array{Float64,2}}
        Array{Array{Float64,2},1}(undef, 0),    # bias::Array{Array{Float64,1}}
        Array{Array{Float64,2},1}(undef, 0),    # delta_w
        Array{Array{Float64,1},1}(undef, 0),    # delta_b
        Array{Array{Float64,2},1}(undef, 0),    # delta_v_w
        Array{Array{Float64,1},1}(undef, 0),    # delta_v_b
        Array{Array{Float64,2},1}(undef, 0),    # delta_s_w
        Array{Array{Float64,1},1}(undef, 0),    # delta_s_b
        Array{Tuple{Int, Int},1}(undef, 0),     # theta_dims::Array{Array{Int64,2}}
        3,                               # output_layer
        Array{Int64,1}(undef, 0),               # layer_units
        ([0.0 0.0], [1.0 0.0])           # norm_factors (mean, std)
    )
end

"""
struct Hyper_parameters holds hyper_parameters used to control training
"""
mutable struct Hyper_parameters          # we will use hp as the struct variable
    units::String               # type of units in all hidden layers -- some day relax this requirement
    alpha::Float64              # learning rate
    lambda::Float64             # L2 regularization rate
    n_hid::Array{Int64,1}       # number of units in each hidden layer
    b1::Float64                 # 1st optimization for momentum or Adam
    b2::Float64                 # 2nd optimization parameter for Adam
    ltl_eps::Float64            # use in denominator with division of very small values to prevent overflow
    alphaovermb::Float64        # calculate outside the learning loop
    do_batch_norm::Bool         # true or false
    norm_mode::String           # "", "none", "standard", or "minmax"
    dropout::Bool               # true or false to choose dropout network
    droplim::Array{Float64,1}   # the probability a node output is kept
    reg::String                 # L2 or "none"
    opt::String                 # Adam or momentum or "none" or "" for optimization
    opt_params::Array{Float64,1}# parameters for optimization
    classify::String            # behavior of output layer: "softmax", "sigmoid", or "regression"
    mb_size::Int64              # minibatch size--calculated; last mini-batch may be smaller
    mb_size_in::Int64           # input of requested minibatch size:  last actual size may be smaller
    n_mb::Int64                 # number of minibatches--calculated
    epochs::Int64               # number of "outer" loops of training
    do_learn_decay::Bool        # step down the learning rate across epochs
    learn_decay::Array{Float64,1}  # reduction factor (fraction) and number of steps
    sparse::Bool
    initializer::String         # "xavier" or "zero"

    Hyper_parameters() = new(       # constructor with defaults--we use hp as the struct variable
        "sigmoid",      # units
        0.35,           # alpha -- OK for nn. way too high for linear regression
        0.01,           # lambda
        [],             # n_hid
        0.9,            # b1
        0.999,          # b2
        1e-8,           # ltl_eps
        0.35,           # alphaovermb -- calculated->not a valid default
        false,          # do_batch_norm
        "none",         # norm_mode
        false,          # dropout
        [0.5],          # droplim
        "L2",           # reg
        "",             # opt
        [],             # opt_params
        "sigmoid",      # classify
        0,              # mb_size
        50,             # mb_size_in  
        100,            # n_mb
        30,             # epochs
        false,          # do_learn_decay
        [1.0, 1.0],     # learn_decay
        false,          # sparse
        "xavier"        # initializer
    )
end


"""
Struct Model_data hold examples and all layer outputs-->
pre-allocate to reduce memory allocations and improve speed
"""
mutable struct Model_data               # we will use train for inputs and test for test data
    # read from training, test, or production data
    inputs #::Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}}   # in_k features by n examples
    targets #::Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}}  # labels for each example
    # calculated in feedforward pass
    a::Array{Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}
    z::Array{Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}
    z_norm::Array{Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}  # same size as z--for batch_norm
    # calculated in backprop (training) pass
    delta_z_norm::Array{Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}   # same size as z
    delta_z::Array{Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}        # same size as z
    grad::Array{Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}
    epsilon::Array{Union{Array{Float64},SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}        # dims of a
    # calculate dropout mask for training
    dropout_random::Array{Array{Float64,2},1}     # randomization for dropout--dims of a
    dropout_mask_units::Array{BitArray{2}, 1}       # boolean filter for dropout--dims of a
    # descriptive
    n::Int64                                  # number of examples
    in_k::Int64                               # number of input features
    out_k::Int64                              # number of output features
    

    Model_data() = new(                 # empty constructor
        zeros(0,0),          # inputs
        zeros(0,0),          # targets
        [zeros(0,0)],       # a
        [zeros(0,0)],       # z
        [zeros(0,0)],       # z_norm -- only pre-allocate if batch_norm
        [zeros(0,0)],       # delta_z_norm
        [zeros(0,0)],       # delta_z
        [zeros(0,0)],       # grad
        [zeros(0,0)],       # epsilon
        Array{Array{Float64,2},1}(undef, 0),   # dropout_random
        Array{BitArray{2},1}(undef, 0),        # dropout_mask_units   Array{Array{Bool,2},1}
        0,                              # n
        0,                              # in_k
        0                               # out_k
    )
end


"""
Struct Training_view holds views on all model data that will be broken into minibatches
"""
mutable struct Training_view               # we will use mb for as the variable for minibatches
    # array of views
    a::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    targets::SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true}
    z::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    z_norm::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    delta_z_norm::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    delta_z::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    grad::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    epsilon::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    dropout_random::Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
    dropout_mask_units::Array{SubArray{Bool,2,BitArray{2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}

    Training_view() = new(                      # empty constructor
        [view(zeros(2,2),:,1:2) for i in 1:2],  # a
        view(zeros(2,2),:,1:2),                 # targets
        [view(zeros(2,2),:,1:2) for i in 1:2],  # z
        [view(zeros(2,2),:,1:2) for i in 1:2],  # z_norm
        [view(zeros(2,2),:,1:2) for i in 1:2],  # delta_z_norm
        [view(zeros(2,2),:,1:2) for i in 1:2],  # delta_z
        [view(zeros(2,2),:,1:2) for i in 1:2],  # grad
        [view(zeros(2,2),:,1:2) for i in 1:2],  # epsilon
        [view(zeros(2,2),:,1:2) for i in 1:2],  # dropout_random
        [view(BitArray([1 1; 1 1]),:,1:2) for i in 1:2]   # dropout_mask_units     
    )

end


"""
struct Batch_norm_params holds batch normalization parameters for
feedfwd calculations and backprop training.
"""
mutable struct Batch_norm_params               # we will use bn as the struct variable
    # learned batch parameters to center and scale data
    gam::Array{Array{Float64,1},1}
    bet::Array{Array{Float64,1},1}
    delta_gam::Array{Array{Float64,1},1}
    delta_bet::Array{Array{Float64,1},1}
    # for standardizing batch values
    mu::Array{Array{Float64,1},1}              # same size as bias = no. of layer units
    stddev::Array{Array{Float64,1},1}          #    ditto
    mu_run::Array{Array{Float64,1},1}          # running average of mu
    std_run::Array{Array{Float64,1},1}         # running average of mu

    Batch_norm_params() = new(           # empty constructor
        Array{Array{Float64,1},1}(undef, 0),    # gam::Array{Array{Float64,1}}
        Array{Array{Float64,1},1}(undef, 0),    # bet::Array{Array{Float64,1}}
        Array{Array{Float64,2},1}(undef, 0),    # delta_gam
        Array{Array{Float64,2},1}(undef, 0),    # delta_bet
        Array{Array{Float64,1},1}(undef, 0),    # mu
        Array{Array{Float64,1},1}(undef, 0),    # stddev
        Array{Array{Float64,1},1}(undef, 0),    # mu_run
        Array{Array{Float64,1},1}(undef, 0)     # std_run
    )
end



# 
# ERROR: MethodError: Cannot `convert` an object of type Array{Float64,2} 
# to an object of type SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true}
# This may have arisen from a call to the constructor SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true}(...),
# since type constructors fall back to convert methods.

# ERROR: MethodError: Cannot `convert` 
# an object of type SubArray{Bool,2,Array{Bool,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true} to an
#    object of type SubArray{Bool,2,BitArray{2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true}
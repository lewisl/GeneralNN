#DONE
#   removed unused array delta_z_scale


#TODO
#   cleaning up batch norm is complicated:
#       affects feedfwd, backprop, pre-allocation (several), momentum, adam search for if [!hp.]*do_batch_norm
#       type dispatch on bn:  either a struct or a bool to eliminate if tests all over to see if we batch normalize
#   don't allocate minibatch if only one--eg, using all of the data--use a pointer to the real data
#       this might work for multiple little minibatches too
#       but for lots of little minibatches, we only need the whole thing for full data predictions
#   check for type stability: @code_warntype pisum(500,10000)
#   is it worth having feedfwdpredict?  along with batchnormfwdpredict?  then no if test since we always know when we are predicting
#   still lots of memory allocations despite the pre-allocation
        # You can devectorize r -= d[j]*A[:,j] with r .= -.(r,d[j]*A[:.j]) 
        #        to get rid of some more temporaries. 
        # As @LutfullahTomak said sum(A[:,j].*r) should devectorize as dot(view(A,:,j),r) 
        #        to get rid of all of the temporaries in there. 
        # To use an infix operator, you can use \cdot, as in view(A,:,j)⋅r.
#   stats on individual regression parameters
#   figure out memory use between train set and minibatch set
#   fix predictions
#   method that uses saved parameters as inputs
#   make affine units a separate layer with functions for feedfwd, gradient and test--do we need to?
#   try "flatscale" x = x / max(x)
#   performance improvements for batch_norm calculations
#   relax minibatch size being exact factor of training data size
#   implement a gradient checking function with option to run it
#   convolutional layers
#   pooling layers
#   better way to handle test not using mini-batches
#   implement early stopping

#   scale weights for cost regularization to accommodate ReLU normalization?
#   separate plots data structure from stats data structure?



"""
Module GeneralNN:

Includes the following functions to run directly:

- train_nn() -- train sigmoid/softmax neural networks for up to 9 hidden layers
- test_score() -- cost and accuracy given test data and saved theta
- save_theta() -- save theta, which is returned by train_nn
- predictions_vector() -- predictions given x and saved theta
- accuracy() -- calculates accuracy of predictions compared to actual labels

To access, run this statement push!(LOAD_PATH, "/Path/To/My/Module/") with the path
to this module.

These data structures are used to hold parameters and data:

- NN_parameters holds theta, bias, delta_w, delta_b, theta_dims, output_layer, layer_units
- Model_data holds inputs, targets, a, z, z_norm, z_scale, epsilon, gradient_function
- Batch_norm_params holds gam (gamma), bet (beta) for batch normalization and Intermediate
data used for training: delta_gam, delta_bet, delta_z_norm, delta_z, mu, stddev, mu_run, std_run

"""
module GeneralNN


# data structures for neural network
export NN_parameters, Model_data, Batch_norm_params, Hyper_parameters

# functions to use
export train_nn, test_score, save_params, load_params, accuracy, predictions_vector, extract_data

using MAT
using JLD
using PyCall
using Plots
pyplot()  # initialize the backend used by Plots
@pyimport seaborn  # prettier charts
# using ImageView    BIG BUG HERE--SEGFAULT--REPORTED

const l_relu_neg = .01  # makes the type constant; value can be changed

"""
struct NN_parameters holds model parameters learned by training and model metadata
"""
mutable struct NN_parameters              # we will use tp as the struct variable
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

    NN_parameters() = new(               # empty constructor
        Array{Array{Float64,2},1}(0),    # theta::Array{Array{Float64,2}}
        Array{Array{Float64,2},1}(0),    # bias::Array{Array{Float64,1}}
        Array{Array{Float64,2},1}(0),    # delta_w
        Array{Array{Float64,1},1}(0),    # delta_b
        Array{Array{Float64,2},1}(0),    # delta_v_w
        Array{Array{Float64,1},1}(0),    # delta_v_b
        Array{Array{Float64,2},1}(0),    # delta_s_w
        Array{Array{Float64,1},1}(0),    # delta_s_b
        Array{Tuple{Int, Int},1}(0),     # theta_dims::Array{Array{Int64,2}}
        2,                               # output_layer
        Array{Int64,1}(0),               # layer_units
        ([0.0 0.0], [1.0 0.0])           # norm_factors (mean, std)
    )
end

"""
struct Hyper_parameters holds hyper_parameters used to control training
"""
mutable struct Hyper_parameters          # we will use hp as the struct variable
    alpha::Float64
    lambda::Float64
    b1::Float64
    b2::Float64
    ltl_eps::Float64
    alphaovermb::Float64
    do_batch_norm::Bool
    norm_mode::String
    droplim::Array{Float64,1}
    reg::String
    opt::String
    classify::String

    Hyper_parameters() = new(       # constructor with defaults--we use hp as the struct variable
        0.35,           # alpha -- OK for nn. way too high for linear regression
        0.01,           # lambda
        0.9,            # b1
        0.999,          # b2
        1e-8,           # ltl_eps
        0.35,           # alphaovermb -- calculated->not a valid default
        false,          # do_batch_norm
        "none",         # norm_mode
        [1.0],          # droplim
        "L2",           # reg
        "",             # opt
        "sigmoid"       # classify
    )
end


"""
Struct Model_data hold examples and all layer outputs-->
pre-allocate to reduce memory allocations and improve speed
"""
mutable struct Model_data               # we will use train for inputs, test for test data and mb for mini-batches
    inputs::Array{Float64,2}            # in_k features by n examples
    targets::Array{Float64,2}           # labels for each example
    a::Array{Array{Float64,2},1}
    z::Array{Array{Float64,2},1}
    z_norm::Array{Array{Float64,2},1}   # same size as z--for batch_norm
    z_scale::Array{Array{Float64,2},1}  # same size as z, often called "y"--for batch_norm
    delta_z_norm::Array{Array{Float64,2},1}    # same size as z
    delta_z::Array{Array{Float64,2},1}         # same size as z
    grad::Array{Array{Float64,2},1}
    epsilon::Array{Array{Float64,2},1}
    drop_ran_w::Array{Array{Float64,2},1} # randomization for dropout--dims of a
    drop_filt_w::Array{Array{Bool,2},1}   # boolean filter for dropout--dims of a
    

    Model_data() = new(                 # empty constructor
        Array{Float64,2}(2,2),          # inputs
        Array{Float64,2}(2,2),          # targets
        Array{Array{Float64,2},1}(0),   # a
        Array{Array{Float64,2},1}(0),   # z
        Array{Array{Float64,2},1}(0),   # z_norm -- only pre-allocate if batch_norm
        Array{Array{Float64,2},1}(0),   # z_scale -- only pre-allocate if batch_norm
        Array{Array{Float64,2},1}(0),    # delta_z_norm
        Array{Array{Float64,2},1}(0),    # delta_z
        Array{Array{Float64,2},1}(0),   # grad
        Array{Array{Float64,2},1}(0),   # epsilon
        Array{Array{Float64,2},1}(0),   # drop_ran_w
        Array{Array{Bool,2},1}(0),      # drop_filt_w
    )
end


"""
Struct Model_view holds views to all model data and all layer outputs-->
pre-allocate to reduce memory allocations and improve speed
"""
mutable struct Model_view               # we will use mb for mini-batches
    a
    z
    z_norm
    z_scale
    delta_z_norm
    delta_z
    grad
    epsilon
    drop_ran_w
    drop_filt_w
    targets
end


"""
struct Batch_norm_params holds batch normalization parameters for
feedfwd calculations and backprop training.
"""
mutable struct Batch_norm_params               # we will use bn as the struct variable
    gam::Array{Array{Float64,1},1}
    bet::Array{Array{Float64,1},1}
    delta_gam::Array{Array{Float64,1},1}
    delta_bet::Array{Array{Float64,1},1}

    mu::Array{Array{Float64,1},1}              # same size as bias = no. of layer units
    stddev::Array{Array{Float64,1},1}          #    ditto
    mu_run::Array{Array{Float64,1},1}
    std_run::Array{Array{Float64,1},1}

    Batch_norm_params() = new(           # empty constructor
        Array{Array{Float64,1},1}(0),    # gam::Array{Array{Float64,1}}
        Array{Array{Float64,1},1}(0),    # bet::Array{Array{Float64,1}}
        Array{Array{Float64,2},1}(0),    # delta_gam
        Array{Array{Float64,2},1}(0),    # delta_bet
        Array{Array{Float64,1},1}(0),    # mu
        Array{Array{Float64,1},1}(0),    # stddev
        Array{Array{Float64,1},1}(0),    # mu_run
        Array{Array{Float64,1},1}(0)     # std_run
    )
end


# """
# Layer template to contain metadata for a single layer above input
# usage (example with 1 hidden layer and softmax output:
#     mylayers = [] # array holds layer dicts for all layers: input=1 to output=n

#     mylayer = deepcopy(layer_template) # do this every time to prevent accidental hold over values!
#     mylayer["kind"] = "input"
#     mylayer["units"] = 40
#     push!(mylayers, mylayer) # absolutely push in order from input to output

#     mylayer = deepcopy(layer_template) # do this every time to prevent accidental hold over values!
#     mylayer["kind"] = "relu"  # note: this is default so could leave as is
#     mylayer["units"] = 10
#     push!(mylayers, mylayer) # absolutely push in order from input to output

#     mylayer = deepcopy(layer_template) # do this every time to prevent accidental hold over values!
#     mylayer["kind"] = "softmax"  # note: this is default so could leave as is
#     mylayer["units"] = 4
#     push!(mylayers, mylayer) # absolutely push in order from input to output

# """
# const layer_template = Dict(
#     "kind" => "relu", # valid: "input", "linear", relu", "sigmoid", "softmax", "l_relu", "conv", "pooling"
#     "units" => 10,
#     "filter_size" => (3,3), # use for conv or pooling
#     "conv_filters" => 6,
#     "pooling_op" => "max", # valid: "max", "avg"
#     "l_relu_neg" => 0.01, # for negative Z values
#     "input_X" => [1.0 2.0 3.0; 4.0 5.0 6.0],  # this seems wrong
#     "input_Y" => [1.0 2.0 3.0]  #  this seems wrong
#     )


"""
function train_nn(matfname::String, epochs::Int64, n_hid::Array{Int64,1}; alpha=0.35,
    mb_size::Int64=0, lambda::Float64=0.01, classify::String="softmax", norm_mode::String="none",
    mom=0.9, units::String="sigmoid", plots::Array{String,1}=["Training", "Learning"])

Train sigmoid/softmax neural networks up to 11 layers.  Detects
number of output labels from data. Detects number of features from data for output units.
Enables any size mini-batch that divides evenly into number of examples.  Plots

    returns:
        NN_parameters  ::= struct that holds all trainable parameters (except...)
        Batch_norm_params ::= struct that holds all batch_norm parameters

    key inputs:
        alpha   ::= learning rate
        lambda  ::= regularization rate
        mb_size ::= mini-batch size=>integer, use 0 to run 1 batch of all examples,
                    otherwise must be an even divisor of the number of examples
        n_hid   ::= array of Int containing number of units in each hidden layer;
                    make sure to use an array even with 1 hidden layer as in [40];
                    use [0] to indicate no hidden layer (typically for linear regression)
        norm_mode  ::= "standard", "minmax" or false => normalize inputs
        do_batch_norm     ::= true or false => normalize each linear layer outputs
        opt            ::= one of "Momentum", "Adam" or "".  default is blank string "".
        opt_params     ::= parameters used by Momentum or Adam
                           Momentum: one floating point value as [.9] (showing default)
                           Adam: 2 floating point values as [.9, .999] (showing defaults)
                           Note that epsilon is ALWAYS set to 1e-8
                           To accept defaults, don't input this parameter or use []
        classify       ::= "softmax", "sigmoid", or "regression" for only the output layer
        units          ::= "sigmoid", "l_relu", "relu" for non-linear activation of all hidden layers
        plots   ::= determines training results collected and plotted
                    any choice of ["Learning", "Cost", "Training", "Test"];
                    for no plots use String[] or ["None"]
        reg     ::= type of regularization, must be one of "L2","Dropout", ""
        droplim ::= array of values between 0.5 and 1.0 determines how much dropout for
                    hidden layers and output layer (ex: [0.8] or [0.8,0.9, 1.0]).  A single
                    value will be applied to all layers.  If fewer values than layers, then the
                    last value extends to remaining layers.


"""
function train_nn(matfname::String, epochs::Int64, n_hid::Array{Int64,1}; alpha::Float64=0.35,
    mb_size::Int64=0, lambda::Float64=0.01, classify::String="softmax", norm_mode::String="none",
    opt::String="", opt_params::Array{Float64,1}=[0.9,0.999], units::String="sigmoid", do_batch_norm::Bool=false,
    reg::String="L2", droplim::Array{Float64,1}=[1.0], plots::Array{String,1}=["Training", "Learning"])

    ################################################################################
    #   This is a front-end function that verifies all inputs and calls run_training
    ################################################################################

    if epochs < 0
        error("Input epochs must be an integer greater than 0")
    end

    # verify number of hidden layers and number of hidden units per layer
    if ndims(n_hid) != 1
        error("Input n_hid must be a vector.")
    elseif size(n_hid,1) > 9
        error("n_hid can only contain 1 to 9 integer values for 1 to 9 hidden layers.")
    elseif isempty(n_hid)  # no hidden layers: user input Int[], which is ok.  [] or Any[] won't work because n_hid must be an array of integers 
        n_hid = Int[]
    elseif n_hid[1] == 0  # no hidden layers: zero has to be the first entry
        n_hid = Int[] # passes all tests and loops to indicate there are no hidden layers
    elseif minimum(n_hid) < 1
        error("Number of hidden units in a layer must be an integer value between 1 and 4096.")
    elseif maximum(n_hid) > 4096
        error("Number of hidden units in a layer must be an integer value between 1 and 4096.")
    end

    if alpha < 0.00001
        warn("Alpha learning rate set too small. Setting to default 0.35")
        alpha = 0.35
    elseif alpha > 1.0
        warn("Alpha learning rate set too large. Setting to defaut 0.35")
        alpha = 0.35
    end

    if mb_size < 0
        error("Input mb_size must be an integer greater or equal to 0")
    end

    classify = lowercase(classify)
    if !in(classify, ["softmax", "sigmoid", "regression"])
        error("classify must be \"softmax\", \"sigmoid\" or \"regression\".")
    end

    units = lowercase(units)
    if !in(units, ["l_relu", "sigmoid", "relu"])
        warn("units must be \"relu,\" \"l_relu,\" or \"sigmoid\". Setting to default \"sigmoid\".")
    end

    if in(units, ["l_relu", "relu"])
        if (norm_mode=="" || norm_mode=="none") && !do_batch_norm
            warn("Better results obtained with relu using input and/or batch normalization. Proceeding...")
        end
    end

    opt = titlecase(lowercase(opt))  # match title case for string argument
    if !in(opt, ["Momentum", "Adam", ""])
        warn("opt must be \"Momentum\" or \"Adam\" or \"\" (nothing).  Setting to \"\" (nothing).")
    end

    reg = titlecase(lowercase(reg))
    if !in(reg, ["L2", "Dropout", ""])
        warn("reg must be \"L2\", \"Dropout\" or \"\" (nothing). Setting to default \"L2\".")
        reg = "L2"
    end

    if reg == "Dropout"
        if !all([(c>=.5 && c<=1.0) for c in droplim])
            error("droplim values must be between 0.5 and 1.0. Quitting.")
        end
    end

    if reg == "L2"
        if lambda < 0.0  # set reg = "" relu with batch_norm
            warn("Lambda regularization rate must be positive floating point value. Setting to 0.01")
            lambda = 0.01
        elseif lambda > 5.0
            warn("Lambda regularization rate set too large. Setting to max of 5.0")
            lambda = 5.0
        end
    end

    valid_plots = ["Training", "Test", "Learning", "Cost", "None", ""]
    plots = titlecase.(lowercase.(plots))  # user input doesn't have use perfect case
    new_plots = [pl for pl in valid_plots if in(pl, plots)] # both valid and input by the user
    if sort(new_plots) != sort(plots) # the plots list included something not in valid_plots
        warn("Plots argument can only include \"Training\", \"Test\", \"Learning\" and \"Cost\" or \"None\" or \"\".
            \nProceeding with default [\"None\"].")
        plots = ["None"]
    else
        plots = new_plots
    end

    run_training(matfname, epochs, n_hid,
        plots=plots, reg=reg, droplim=droplim, alpha=alpha, mb_size=mb_size, lambda=lambda,
        opt=opt, opt_params=opt_params, classify=classify,
        norm_mode=norm_mode, do_batch_norm=do_batch_norm, units=units);
end


function run_training(matfname::String, epochs::Int64, n_hid::Array{Int64,1};
    plots::Array{String,1}=["Training", "Learning"], reg="L2", droplim=[1.0], alpha=0.35,
    mb_size=0, lambda=0.01, opt="", opt_params=[],
    classify="softmax", norm_mode="none", do_batch_norm=false, units="sigmoid")

    # start the cpu clock
    tic()

    # seed random number generator.  For runs of identical models the same weight initialization
    # will be used, given the number of parameters to be estimated.  Enables better comparisons.
    srand(70653)  # seed int value is meaningless


    ##########################################################
    #   pre-allocate and initialize variables
    ##########################################################

    # create data containers
    train = Model_data()  # train holds all the data and layer inputs/outputs
    test = Model_data()

    # load training data and test data (if any)
    train.inputs, train.targets, test.inputs, test.targets, norm_factors = extract_data(matfname, norm_mode)
    # debug
    # println("norm_factors ", typeof(norm_factors))
    # println(norm_factors)

    # set some useful variables
    in_k,n = size(train.inputs)  # number of features in_k (rows) by no. of examples n (columns)
    out_k = size(train.targets,1)  # number of output units
    dotest = size(test.inputs, 1) > 0  # there is testing data

    #setup mini-batch
    if mb_size < 1
        mb_size = n  # use 1 (mini-)batch with all of the examples
    elseif mb_size > n
        mb_size = n
    elseif mod(n, mb_size) != 0
        error("Mini-batch size $mb_size does not divide evenly into samples $n.")
    end
    n_mb = Int(n / mb_size)  # number of mini-batches

    if mb_size < n
        # randomize order of all training samples:
            # labels in training data often in a block, which will make
            # mini-batch train badly because a batch will not contain mix of target labels
        select_index = randperm(n)
        train.inputs[:] = train.inputs[:, select_index]
        train.targets[:] = train.targets[:, select_index]
    end

    # set Hyper_parameters to provided inputs
    hp = Hyper_parameters()  #  sets the defaults!
        hp.alpha =  alpha
        hp.lambda = lambda
        hp.reg = reg
        hp.classify = classify

    # no batch normalization for 1 batch
    hp.do_batch_norm = n_mb == 1 ? false : do_batch_norm  
    hp.alphaovermb = alpha / mb_size  # calc once, use in loop

    # set parameters for Momentum or Adam optimization
    if opt == "Momentum" || opt == "Adam"
        hp.opt = opt
        if !(opt_params == [])  # use inputs for opt_params
            # set b1 for Momentum and Adam
            if opt_params[1] > 1.0 || opt_params[1] < 0.5
                warn("first opt_params for Momentum or Adam should be between 0.5 and 0.999. Using default")
                # nothing to do:  hp.b1 = 0.9 and hp.b2 = 0.999 and hp.ltl_eps = 1e-8
            else
                hp.b1 = opt_params[1] # use the passed parameter
            end
            # set b2 for Adam
            if length(opt_params) > 1
                if opt_params[2] > 1.0 || opt_params[1] < 0.9
                    warn("second opt_params for Adam should be between 0.9 and 0.999. Using default")
                else
                    hp.b2 = opt_params[2]
                end
            end
        end
    else
        hp.opt = ""
    end

    # DEBUG see if hyper_parameters set correctly
    # for sym in fieldnames(hp)
    #    println(sym," ",getfield(hp,sym), " ", typeof(getfield(hp,sym)))
    # end

    # statistics for plots and history data
    plotdef = setup_plots(epochs, dotest, plots)

    ##########################################################################
    #  pre-allocate variables
    ##########################################################################

    # neural net model parameters
    tp = NN_parameters()  # trainedparameters: tp holds all the parameters that will be trained and some metadata
    preallocate_nn_params!(tp, hp, n_hid, in_k, n, out_k)
    tp.norm_factors = norm_factors

    # feedfwd training data
    preallocate_feedfwd!(train, tp, n, hp.do_batch_norm)

    # feedfwd test data--if inputs are not all zeros
    if dotest
        testn = size(test.inputs,2)
        preallocate_feedfwd!(test, tp, testn, hp.do_batch_norm)
    else
        testn = 0
    end

    # pre-allocate feedfwd mini-batch inputs and targets
    mb = Model_data()  # mb holds all layer data for mini-batches
    preallocate_minibatch!(mb, tp, mb_size, in_k, n, out_k, hp.do_batch_norm)
    create_minibatch_views!(train, tp, hp, mb_size)

    # batch normalization parameters
    bn = Batch_norm_params()  # do we always need the data structure to run?  yes--TODO fix this
    if hp.do_batch_norm
        preallocate_batchnorm!(bn, mb, tp.layer_units)
    end

    # dropout parameters: droplim is in hp (Hyper_parameters),
    #    drop_ran_w and drop_filt_w are in mb or train (Model_data)
    # set a droplim for each layer (input layer will be ignored)
    if hp.reg == "Dropout"
        hp.droplim = droplim
        # fill droplim to match number of layers
        if length(droplim) > length(tp.layer_units)
            hp.droplim = droplim[1:tp.output_layer] # truncate
        elseif length(droplim) < length(tp.layer_units)
            for i = 1:length(tp.layer_units)-length(droplim)
                push!(hp.droplim,droplim[end]) # pad
            end
        end
        mb.drop_ran_w = deepcopy(mb.a)
        push!(mb.drop_filt_w,fill(true,(2,2))) # for input layer, not used
        for item in mb.drop_ran_w[2:end]
            push!(mb.drop_filt_w,fill(true,size(item)))
        end
    end

    # debug
    # verify correct dimensions of dropout filter
    # for item in mb.drop_filt_w
    #     println(size(item))
    # end
    # error("that's all folks!....")

    #################################################################
    #   define and choose functions to be used in neural net training
    #################################################################

    if units == "sigmoid"
        unit_function! = sigmoid!
        do_batch_norm = false
    elseif units == "l_relu"
        unit_function! = l_relu!
    elseif units == "relu"
        unit_function! = relu!
    end

    if unit_function! == sigmoid!
        gradient_function! = sigmoid_gradient!
    elseif unit_function! == l_relu!
        gradient_function! = l_relu_gradient!
    elseif unit_function! == relu!
        gradient_function! = relu_gradient!
    end

    if out_k > 1  # more than one output (unit)
        if classify == "sigmoid"
            classify_function! = sigmoid!
        elseif classify == "softmax"
            classify_function! = softmax!
        else
            error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
        end
    else
        if classify == "sigmoid"
            classify_function! = sigmoid!  # for one output label
        elseif classify == "regression"
            classify_function! = regression!
        else
            error("Function to classify output must be \"sigmoid\" or \"regression\".")
        end
    end

    # set cost function
    cost_function = classify=="regression" ? mse_cost : cross_entropy_cost

    # function lists to be passed to feedfwd! and backprop!
    fwd_functions = (batch_norm_fwd!, unit_function!, classify_function!)
    back_functions = (batch_norm_back!, gradient_function!)


    ##########################################################
    #   neural network training loop
    ##########################################################

    t = 0

    # debug
    # println(train.inputs)
    # println(train.targets)

    for ep_i = 1:epochs  # loop for "epochs" with counter epoch i as ep_i
        for mb_j = 1:n_mb  # loop for mini-batches with counter minibatch j as mb_j

            first_example = (mb_j - 1) * mb_size + 1  # mini-batch subset for the inputs->layer 1
            last_example = first_example + mb_size - 1

            t += 1  # update counter

            mb.a[1][:] = train.inputs[:,first_example:last_example] # m-b input layer activation
            mb.targets[:] = train.targets[:, first_example:last_example]

            feedfwd!(tp, bn, mb, fwd_functions, hp)  # for all layers

            # debug
            # println(mb.a[2])

            backprop!(tp, bn, mb, back_functions, hp, t)  # for all layers

            if hp.opt == "Momentum"
                momentum!(tp, hp)  # for all hidden layers
            elseif hp.opt == "Adam"
                adam!(tp, hp, t)  # for all hidden layers
            end

            # update weights, bias, and batch_norm parameters
            @fastmath for hl = 2:tp.output_layer        # hl iterates over each hidden layer and the output layer

                # update weights
                tp.theta[hl] .= tp.theta[hl] .- (hp.alphaovermb .* tp.delta_w[hl])
                if hp.reg == "L2"  # subtract regularization term
                    tp.theta[hl] .= tp.theta[hl] .- (hp.alphaovermb .* (hp.lambda .* tp.theta[hl]))
                end

                #update bias
                if !hp.do_batch_norm
                    tp.bias[hl] .= tp.bias[hl] .- (hp.alphaovermb .* tp.delta_b[hl])
                end

                # update batch normalization parameters
                if hp.do_batch_norm
                    bn.gam[hl][:] -= hp.alphaovermb .* bn.delta_gam[hl]
                    bn.bet[hl][:] -= hp.alphaovermb .* bn.delta_bet[hl]
                end

            # debug
            # println(tp.theta)
            end  # layer loop
        end # mini-batch loop

        # stats for all mini-batches of one epoch
        gather_stats!(ep_i, plotdef, train, test, tp, bn, cost_function, fwd_functions, n, testn, hp)  # n or mb_size

    end # epoch loop

    #####################################################################
    # print and plot training statistics after all epochs
    #####################################################################

    println("Training time: ",toq()," seconds")  # cpu time since tic() =>  toq() returns secs without printing

    feedfwd!(tp, bn, train, fwd_functions, hp, istrain=false)  # output for entire training set
    println("Fraction correct labels predicted training: ",
            hp.classify == "regression" ? r_squared(train.targets, train.a[tp.output_layer])
                : accuracy(train.targets, train.a[tp.output_layer],epochs))
    println("Final cost training: ", cost_function(train.targets, train.a[tp.output_layer], n,
                    tp.theta, hp, tp.output_layer))

    # output test statistics
    if dotest
        feedfwd!(tp, bn, test, fwd_functions, hp, istrain=false)
        println("Fraction correct labels predicted test: ",
                hp.classify == "regression" ? r_squared(test.targets, test.a[tp.output_layer])
                    : accuracy(test.targets, test.a[tp.output_layer],epochs))
        println("Final cost test: ", cost_function(test.targets, test.a[tp.output_layer], testn,
            tp.theta, hp, tp.output_layer))
    end

    # output improvement of last 10 iterations for test data
    if plotdef["plot_switch"]["Test"]
        if plotdef["plot_switch"]["Learning"]
            println("Test data accuracy in final 10 iterations:")
            printdata = plotdef["fracright_history"][end-10+1:end, plotdef["col_test"]]
            for i=1:10
                @printf("%0.3f : ", printdata[i])
            end
            print("\n")
        end
    end

    # plot the progress of cost and/or learning accuracy
    plot_output(plotdef)

    return train, tp, bn, hp;  # trained model parameters, batch_norm parameters, hyper parameters

end  # function run_training


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
function extract_data(matfname::String, norm_mode::String="none")
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

    if lowercase(norm_mode) == "standard"
        # normalize training data
        x_mu = mean(inputs, 2)
        x_std = std(inputs, 2)
        inputs = (inputs .- x_mu) ./ (x_std .+ 1e-08)
        # normalize test data
        if in("test", keys(df))
            test_inputs = (test_inputs .- x_mu) ./ (x_std .+ 1e-08)
        end
        norm_factors = (x_mu, x_std)
    elseif lowercase(norm_mode) == "minmax"
        # normalize training data
        x_max = maximum(inputs, 2)
        x_min = minimum(inputs, 2)
        inputs = (inputs .- x_min) ./ (x_max .- x_min .+ 1e-08)
        # normalize test data
        if in("test", keys(df))
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

    return inputs, targets, test_inputs, test_targets, norm_factors
end


####################################################################
#  functions to pre-allocate data updated during training loop
####################################################################

function preallocate_feedfwd!(dat, tp, n, do_batch_norm)
    # we don't pre-allocate epsilon, grad used during backprop
    dat.a = [dat.inputs]
    dat.z = [zeros(2,2)] # not used for input layer
    for i = 2:tp.output_layer-1  # hidden layers
        push!(dat.z, zeros(tp.layer_units[i], n))
        push!(dat.a, zeros(size(dat.z[i])))  #  and up...  ...output layer set after loop
    end
    push!(dat.z, zeros(size(tp.theta[tp.output_layer],1),n))
    push!(dat.a, zeros(size(tp.theta[tp.output_layer],1),n))

    if do_batch_norm  # required for full pass performance stats
        dat.z_norm = deepcopy(dat.z)
        dat.z_scale = deepcopy(dat.z)  # same size as z, often called "y"
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
    if hp.opt == "Momentum" || hp.opt == "Adam"
        tp.delta_v_w = [zeros(a) for a in tp.delta_w]
        tp.delta_v_b = [zeros(a) for a in tp.delta_b]
    end
    if hp.opt == "Adam"
        tp.delta_s_w = [zeros(a) for a in tp.delta_w]
        tp.delta_s_b = [zeros(a) for a in tp.delta_b]
    end


end


function preallocate_minibatch!(mb, tp, mb_size, in_k, n, out_k, do_batch_norm)
    # mb.inputs = zeros(in_k, mb_size)  # only used for pre-allocation
    mb.inputs = Array{Float64,2}(in_k, mb_size)
    mb.targets = zeros(out_k, mb_size)
    preallocate_feedfwd!(mb, tp, mb_size, do_batch_norm)
    mb.epsilon = deepcopy(mb.a)  # looks like activations of each unit above input layer
    mb.grad = deepcopy(mb.z)
    if do_batch_norm
        mb.z_norm = deepcopy(mb.z)
        mb.z_scale = deepcopy(mb.z)
    end
end


function create_minibatch_views!(train, mb, tp, hp, mb_size)
    n_layers = tp.output_layer

    v_a = [view(train.a[i],:,1:1) for i = 1:n_layers]
    v_z = [view(train.z[i],:,1:1) for i = 1:n_layers]

    # used only in backprop--"real" arrays not views because training data doesn't have these
    v_grad = [zeros(size(v_z[i],1), mb_size) for i = 1:n_layers]
    v_epsilon = [zeros(size(v_a[i], 1), mb_size) for i = 1:n_layers]

    # used only if dropout
    if hp.reg == "dropout"
        v_drop_ran_w = deepcopy(v_epsilon)  # size of physical a
        v_drop_filt_w = [fill(true,size(v_epsilon[i])) for i = 1:n_layers]
    else
        v_drop_ran_w = []
        v_drop_filt_w = []
    end

    if hp.do_batch_norm
        v_z_norm = [view(train.z_norm[i],:,1:1) for i = 1:n_layers]
        v_z_scale = [view(train.z_scale[i],:,1:1) for i = 1:n_layers]
        v_delta_z_norm = [view(train.delta_z_norm[i],:,1:1) for i = 1:n_layers]
        v_delta_z = [view(train.delta_z[i],:,1:1) for i = 1:n_layers]
    else
        v_z_norm = [view(train.z_norm,:,1:1)]
        v_z_scale = [view(train.z_scale,:,1:1)]
        v_delta_z_norm = [view(train.delta_z_norm,:,1:1)]
        v_delta_z = [view(train.delta_z,:,1:1)]
    end

    mb = GeneralNN.Model_view(
        v_a,
        v_z,
        v_z_norm,
        v_z_scale,
        v_delta_z_norm,
        v_delta_z,
        v_grad,
        v_epsilon,
        v_drop_ran_w,
        v_drop_filt_w,
        v_a[tp.output_layer]   # view of targets = activation of the output layer  
    )

    return mb
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

    # part of Model_data--but only used if batch normalization and only for minibatches
    mb.delta_z = deepcopy(mb.z_norm)
    mb.delta_z_norm = deepcopy(mb.z_norm)
end

###########################################################################

"""
function feedfwd!(tp, bn, dat, fwd_functions, do_batch_norm; istrain)
    modifies a, a_wb, z in place to reduce memory allocations
    send it all of the data or a mini-batch

    feed forward from inputs to output layer predictions
"""
function feedfwd!(tp, bn, dat, fwd_functions, hp; istrain=true)

    (batch_norm_fwd!, unit_function!, classify_function!) = fwd_functions

    @fastmath for hl = 2:tp.output_layer-1  # hidden layers

        if hp.do_batch_norm
            dat.z[hl][:] = tp.theta[hl] * dat.a[hl-1]  # linear with no bias
            batch_norm_fwd!(bn, dat, hl, istrain)
            unit_function!(dat.z_scale[hl],dat.a[hl]) # non-linear function
            if istrain && hp.reg == "Dropout"
                dropout!(dat,hp,hl)
            end
        else
            dat.z[hl][:] = tp.theta[hl] * dat.a[hl-1] .+ tp.bias[hl]  # linear with bias
            unit_function!(dat.z[hl],dat.a[hl])
        end

    end

    @fastmath dat.z[tp.output_layer][:] = (tp.theta[tp.output_layer] * dat.a[tp.output_layer-1]
        .+ tp.bias[tp.output_layer])  # TODO use bias in the output layer with no batch norm?

    classify_function!(dat.z[tp.output_layer], dat.a[tp.output_layer])  # a = activations = predictions

end


"""
function backprop!(tp, dat, back_functions, do_batch_norm)
    Argument tp.delta_w holds the computed gradients for weights, delta_b for bias
    Modifies dat.epsilon, tp.delta_w, tp.delta_b in place--caller uses tp.delta_w, tp.delta_b
    Use for training iterations
    Send it all of the data or a mini-batch
    Intermediate storage of dat.a, dat.z, dat.epsilon, tp.delta_w, tp.delta_b reduces memory allocations
"""
function backprop!(tp, bn, dat, back_functions, hp, t)

    (batch_norm_back!, gradient_function!) = back_functions

    dat.epsilon[tp.output_layer][:] = dat.a[tp.output_layer] .- dat.targets
    @fastmath tp.delta_w[tp.output_layer][:] = dat.epsilon[tp.output_layer] * dat.a[tp.output_layer-1]' # 2nd term is effectively the grad for mse
    @fastmath tp.delta_b[tp.output_layer][:] = sum(dat.epsilon[tp.output_layer],2)

    @fastmath for hl = (tp.output_layer - 1):-1:2  # loop over hidden layers
        if hp.do_batch_norm
            gradient_function!(dat.z_scale[hl], dat.grad[hl])
            dat.epsilon[hl][:] = tp.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl]
            batch_norm_back!(tp, dat, bn, hl)
            tp.delta_w[hl][:] = dat.delta_z[hl] * dat.a[hl-1]'
        else
            gradient_function!(dat.z[hl], dat.grad[hl])
            dat.epsilon[hl][:] = tp.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl]
            tp.delta_w[hl][:] = dat.epsilon[hl] * dat.a[hl-1]'
            tp.delta_b[hl][:] = sum(dat.epsilon[hl],2)  #  times a column of 1's = sum(row)
        end

    end

end


function cross_entropy_cost(targets, predictions, n, theta, hp, output_layer)
    # n is count of all samples in data set--use with regularization term
    # mb_size is count of all samples used in training batch--use with cost
    # these may be equal
    cost = (-1.0 / n) * (dot(targets,log.(predictions .+ 1e-50)) +
        dot((1.0 .- targets), log.(1.0 .- predictions .+ 1e-50)))

    # if isnan(cost)
    #    println("predictions <= 0? ",any(predictions .== 0.0))
    #    println("log 1 - predictions? ", any(isinf.(log.(1.0 .- predictions))))
    #    error("problem with cost function")
    # end
    @fastmath if hp.reg == "L2"  # set reg="" if not using regularization
        # regterm = hp.lambda/(2.0 * n) .* sum([sum(th .* th) for th in theta[2:output_layer]])
        regterm = hp.lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end


function mse_cost(targets, predictions, n, theta, hp, output_layer)
    cost = (1.0 / (2.0 * n)) .* sum((targets .- predictions) .^ 2.0)
    @fastmath if hp.reg == "L2"  # set reg="" if not using regularization
        # regterm = hp.lambda/(2.0 * n) .* sum([sum(th .* th) for th in theta[2:output_layer]])
        regterm = hp.lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end


# not using yet
function mse_grad!(mb, layer) # only for output layer using mse_cost
    # do we really need this?

    mb.grad[layer] = mb.a[layer-1]

end


function momentum!(tp, hp)
    @fastmath for hl = (tp.output_layer - 1):-1:2  # loop over hidden layers
        tp.delta_v_w[hl] .= hp.b1 .* tp.delta_v_w[hl] .+ (1.0 - hp.b1) .* tp.delta_w[hl]
        tp.delta_w[hl] .= tp.delta_v_w[hl]

        if !hp.do_batch_norm  # then we need to do bias term
            tp.delta_v_b[hl] .= hp.b1 .* tp.delta_v_b[hl] .+ (1.0 - hp.b1) .* tp.delta_b[hl]
            tp.delta_b[hl] .= tp.delta_v_b[hl]
        end
    end
end

function adam!(tp, hp, t)

    @fastmath for hl = (tp.output_layer - 1):-1:2  # loop over hidden layers
        tp.delta_v_w[hl] .= hp.b1 .* tp.delta_v_w[hl] .+ (1.0 - hp.b1) .* tp.delta_w[hl]
        tp.delta_s_w[hl] .= hp.b2 .* tp.delta_s_w[hl] .+ (1.0 - hp.b2) .* tp.delta_w[hl].^2
        tp.delta_w[hl] .= (  (tp.delta_v_w[hl] ./ (1.0 - hp.b1^t)) ./
                              (sqrt.(tp.delta_s_w[hl] ./ (1.0 - hp.b2^t)) + hp.ltl_eps)  )

        if !hp.do_batch_norm  # then we need to do bias term
            tp.delta_v_b[hl] .= hp.b1 .* tp.delta_v_b[hl] .+ (1.0 - hp.b1) .* tp.delta_b[hl]
            tp.delta_s_b[hl] .= hp.b2 .* tp.delta_s_b[hl] .+ (1.0 - hp.b2) .* tp.delta_b[hl].^2
            tp.delta_b[hl] .= (  (tp.delta_v_b[hl] ./ (1.0 - hp.b1^t)) ./
                              (sqrt.(tp.delta_s_b[hl] ./ (1.0 - hp.b2^t)) + hp.ltl_eps)  )
        end
    end
end

# TODO will this work with batch norm?
# this must be done one layer at a time--because of layer dependence
function dropout!(dat,hp,hl)
    dat.drop_ran_w[hl] .= rand(size(dat.drop_ran_w[hl]))
    dat.drop_filt_w[hl] .= dat.drop_ran_w[hl] .< hp.droplim[hl]
    dat.a[hl] .*= dat.drop_filt_w[hl]
    dat.a[hl] ./= hp.droplim[hl]
end


###########################################################################
#  layer functions:  activation and gradients for units of different types
###########################################################################

# two methods for linear layer units, with bias and without
function affine(weights, data, bias)  # with bias
    return weights * data .+ bias
end


function affine(weights, data)  # no bias
    return weights * data
end


function sigmoid!(z::Array{Float64,2}, a::Array{Float64,2})
    a[:] = 1.0 ./ (1.0 .+ exp.(-z))
end


function l_relu!(z::Array{Float64,2}, a::Array{Float64,2}) # leaky relu
    a[:] = map(j -> j >= 0.0 ? j : l_relu_neg * j, z)
end


function relu!(z::Array{Float64,2}, a::Array{Float64,2})
    a[:] = max.(z, 0.0)
end


function softmax!(z::Array{Float64,2}, a::Array{Float64,2})

    expf = similar(a)
    expf[:] = exp.(z .- maximum(z,1))
    a[:] = expf ./ sum(expf, 1)

end


function regression!(z::Array{Float64,2}, a::Array{Float64,2})
    a[:] = z[:]
end


# two methods for gradient of linear layer units:  without bias and with
# not using this yet
function affine_gradient(data, layer)  # no bias
    return data.a[layer-1]'
end


function sigmoid_gradient!(z::Array{Float64,2}, grad::Array{Float64,2})
    sigmoid!(z, grad)
    grad[:] = grad .* (1.0 .- grad)
end


function l_relu_gradient!(z::Array{Float64,2}, grad::Array{Float64,2})
    grad[:] = map(j -> j > 0.0 ? 1.0 : l_relu_neg, z);
end


function relu_gradient!(z::Array{Float64,2}, grad::Array{Float64,2})
    grad[:] = map(j -> j > 0.0 ? 1.0 : 0.0, z);
end


###########################################################################

function batch_norm_fwd!(bn, dat, hl, istrain=true)
    in_k,mb = size(dat.z[hl])
    if istrain
        bn.mu[hl][:] = mean(dat.z[hl], 2)          # use in backprop
        bn.stddev[hl][:] = std(dat.z[hl], 2)
        dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu[hl]) ./ bn.stddev[hl]  # normalized: 'aka' xhat or zhat
        dat.z_scale[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: 'aka' y
        bn.mu_run[hl][:] = (  bn.mu_run[hl][1] == 0.0 ? bn.mu[hl] :
            0.9 .* bn.mu_run[hl] .+ 0.1 .* bn.mu[hl]  )
        bn.std_run[hl][:] = (  bn.std_run[hl][1] == 0.0 ? bn.stddev[hl] :
            0.9 .* bn.std_run[hl] + 0.1 .* bn.stddev[hl]  )
    else  # predictions with existing parameters
        dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu_run[hl]) ./ bn.std_run[hl]  # normalized: 'aka' xhat or zhat
        dat.z_scale[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: 'aka' y
    end
end


function batch_norm_back!(tp, dat, bn, hl)
    d,mb = size(dat.epsilon[hl])
    bn.delta_bet[hl][:] = sum(dat.epsilon[hl], 2)
    bn.delta_gam[hl][:] = sum(dat.epsilon[hl] .* dat.z_norm[hl], 2)
    dat.delta_z_norm[hl][:] = bn.gam[hl] .* dat.epsilon[hl]
    dat.delta_z[hl][:] = (
        (1.0 / mb) .* (1.0 ./ bn.stddev[hl]) .* (
            mb .* dat.delta_z_norm[hl] .- sum(dat.delta_z_norm[hl],2) .-
            dat.z_norm[hl] .* sum(dat.delta_z_norm[hl] .* dat.z_norm[hl], 2)
            )
        )
end


function plot_output(plotdef)
    # plot the progress of training cost and/or learning
    if (plotdef["plot_switch"]["Training"] || plotdef["plot_switch"]["Test"])

        if plotdef["plot_switch"]["Cost"]
            plt_cost = plot(plotdef["cost_history"], title="Cost Function",
                labels=plotdef["plot_labels"], ylims=(0.0, Inf))
            display(plt_cost)  # or can use gui()
        end

        if plotdef["plot_switch"]["Learning"]
            plt_learning = plot(plotdef["fracright_history"], title="Learning Progress",
                labels=plotdef["plot_labels"], ylims=(0.0, 1.0), reuse=false)
                # reuse=false  open a new plot window
            display(plt_learning)
        end

        if (plotdef["plot_switch"]["Cost"] || plotdef["plot_switch"]["Learning"])
            println("Press enter to close plot window..."); readline()
            closeall()
        end
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


function gather_stats!(i, plotdef, train, test, tp, bn, cost_function, fwd_functions, train_n, testn, hp)

    if plotdef["plot_switch"]["Training"]
        feedfwd!(tp, bn, train, fwd_functions, hp, istrain=false)
        if plotdef["plot_switch"]["Cost"]
            plotdef["cost_history"][i, plotdef["col_train"]] = cost_function(train.targets,
                train.a[tp.output_layer], train_n, tp.theta, hp, tp.output_layer)
        end
        if plotdef["plot_switch"]["Learning"]
            plotdef["fracright_history"][i, plotdef["col_train"]] = (  hp.classify == "regression"
                    ? r_squared(train.targets, train.a[tp.output_layer])
                    : accuracy(train.targets, train.a[tp.output_layer], i)  )
        end
    end

    if plotdef["plot_switch"]["Test"]
        feedfwd!(tp, bn, test, fwd_functions, hp, istrain=false)
        if plotdef["plot_switch"]["Cost"]
            cost = cost_function(test.targets,
                test.a[tp.output_layer], testn, tp.theta, hp, tp.output_layer)
                # println("iter: ", i, " ", "cost: ", cost)
            plotdef["cost_history"][i, plotdef["col_test"]] =cost
        end
        if plotdef["plot_switch"]["Learning"]
            # printdims(Dict("test.a"=>test.a, "test.z"=>test.z))
            plotdef["fracright_history"][i, plotdef["col_test"]] = (  hp.classify == "regression"
                    ? r_squared(test.targets, test.a[tp.output_layer])
                    : accuracy(test.targets, test.a[tp.output_layer], i)  )
        end
    end

    # println("train 1 - predictions:")
    # println(1.0 .- train.a[tp.output_layer][:, 1:2])

    # println("test 1 - predictions:")
    # println(1.0 .- test.a[tp.output_layer][:, 1:2])
    


end


function accuracy(targets, preds, i)
    if size(targets,1) > 1
        targetmax = ind2sub(size(targets),vec(findmax(targets,1)[2]))[1]
        predmax = ind2sub(size(preds),vec(findmax(preds,1)[2]))[1]
        try
            fracright = mean([ii ? 1.0 : 0.0 for ii in (targetmax .== predmax)])
        catch
            println("iteration:      ", i)
            println("targetmax size  ", size(targetmax))
            println("predmax size    ", size(predmax))
            println("targets in size ", size(targets))
            println("preds in size   ", size(preds))
        end
    else
        # works because single output unit is sigmoid
        choices = [j >= 0.5 ? 1.0 : 0.0 for j in preds]
        fracright = mean(convert(Array{Int},choices .== targets))
    end
    return fracright
end


function r_squared(targets, preds)
    ybar = mean(targets)
    return 1.0 - sum((targets .- preds).^2.) / sum((targets .- ybar).^2.)
end


"""

    function save_params(jld_fname, tp, bn, hp)

Save the trained parameters: tp, batch_norm parameters: bn, and hyper parameters: hp,
as a JLD file.

Can be used to run the model on prediction data or to evaluate other
test data results (cost and accuracy).
"""
function save_params(jld_fname, tp, bn, hp )
    # check if output file exists and ask permission to overwrite
    if isfile(jld_fname)
        print("Output file $jld_fname exists. OK to overwrite? ")
        resp = readline()
        if contains(lowercase(resp), "y")
            rm(jld_fname)
        else
            error("File exists. Replied no to overwrite. Quitting.")
        end
    end

    # to translate to unnormalized regression coefficients: m = mhat / stdx, b = bhat - (m*xmu)

    # write the JLD formatted file (based on hdf5)
    jldopen(jld_fname, "w") do f
        write(f, "tp", tp)
        write(f, "hp", hp)
        write(f, "bn", bn)
    end

end


"""

    function load_params(jld_fname)

Load the trained parameters: tp, batch_norm parameters: bn, and hyper parameters: hp,
from a JLD file.

Can be used to run the model on prediction data or to evaluate other
test data results (cost and accuracy).

returns: tp, bn, hp
These are mutable structs.  Use fieldnames(tp) to list the fields.
"""
function load_params(jld_fname)
    jldopen(jld_fname, "r") do f
        tp = read(f, "tp")
        bn = read(f, "bn")
        hp = read(f, "hp")
    end
    return tp,bn,hp
end


"""

    function test_score(theta_fname, data_fname, lambda = 0.01,
        classify="softmax")

Calculate the test accuracy score and cost for a dataset containing test or validation
data.  This data must contain outcome labels, e.g.--y.
"""
function test_score(theta_fname, data_fname, lambda = 0.01, classify="softmax")
    # read theta
    dtheta = matread(theta_fname)
    theta = dtheta["theta"]

    # read the test data:  can be in a "test" key with x and y or can be top-level keys x and y
    df = matread(data_fname)

    if in("test", keys(df))
        inputs = df["test"]["x"]'  # set examples as columns to optimize for julia column-dominant operations
        targets = df["test"]["y"]'
    else
        inputs = df["x"]'
        targets = df["y"]'
    end
    n = size(inputs,2)
    output_layer = size(theta,1)

    # setup cost
    cost_function = cross_entropy_cost

    predictions = predict(inputs, theta)
    score = accuracy(targets, predictions)
    println("Fraction correct labels predicted test: ", score)
    println("Final cost test: ", cost_function(targets, predictions, n, theta, hp, output_layer))

    return score
end


"""

    function predictions_vector(theta_fname, data_fname, lambda = 0.01,
        classify="softmax")

    returns vector of all predictions

Return predicted values given inputs and theta.  Not used by training.
Use when theta is already trained and saved to make predictions
for input operational data to use your model. Resolves sigmoid or softmax outputs
to (zero, one) values for each output unit.

Simply does feedforward, but does some data staging first.
"""
function predictions_vector(theta_fname, data_fname, lambda = 0.01, classify="softmax")
  # read theta
    dtheta = matread(theta_fname)
    theta = dtheta["theta"]

    # read the operational data:  can be in a "train" key with x or can be top-level key x
    df = matread(data_fname)

    if in("train", keys(df))
        inputs = df["train"]["x"]'  # set examples as columns to optimize for julia column-dominant operations
    else
        inputs = df["x"]'
    end
    n = size(inputs,2)
    output_layer = size(theta,1)

    predictions = predict(inputs, theta)
    if size(predictions,1) > 1
        # works for output units sigmoid or softmax
        ret = [indmax(predictions[:,i]) for i in 1:size(predictions,2)]
    else
        # works because single output unit is sigmoid
        ret = [j >= 0.5 ? 1.0 : 0.0 for j in predictions]
    end

    return ret
end


# TODO -- THIS IS BROKEN!
"""

    function predict(inputs, theta)

Generate predictions given theta and inputs.
Not suitable in a loop because of all the additional allocations.
Use with one-off needs like scoring a test data set or
producing predictions for operational data fed into an existing model.
"""
function predict(inputs, theta)   ### TODO this is seriously broken!
    # set some useful variables
    in_k,n = size(inputs) # number of features in_k by no. of examples n
    output_layer = size(theta,1)
    out_k = size(theta[output_layer],1)  # number of output units or

    # setup cost
    cost_function = cross_entropy_cost

    # setup class function
    if t > 1  # more than one output (unit)
        if classify == "sigmoid"
            classify_function! = sigmoid!
        elseif classify == "softmax"
            classify_function! = softmax!
        else
            error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
        end
    else
        classify_function! = sigmoid!  # for one output (unit)
    end

    a_test,  z_test = preallocate_feedfwd(inputs, tp, n)
    predictions = feedfwd!(tp, bn, dat, fwd_functions, do_batch_norm, istrain=false)
end


function printby2(nby2)  # not used currently
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end


"""Print the sizes of matrices you pass in as Dict("name"=>var,...)"""
function printdims(indict)
    n = length(indict)
    println("\nSizes of matrices\n")
    for (n, item) in indict
        println(n,": ",size(item))
    end
    println("")
end

end # module GeneralNN
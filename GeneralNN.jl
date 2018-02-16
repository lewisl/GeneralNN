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
data used for training: delta_gam, delta_bet, delta_z_norm, delta_z_scale,
delta_z, mu, stddev, mu_run, std_run

"""
module GeneralNN



#DONE



#TODO
#   use one function with two methods for weight_update to distinguish with and without regularization
#   make affine units a separate layer_template
#   relax minibatch size being exact factor of training data size
#   implement dropout
#   set specific initialization for sigmoid units
#   verify that bias is not used for batch_norm
#   put metaparameters in NN_parameters
#   implement a gradient checking function with option to run it
#   fix cost calculation: sometimes results in NaN with relu and batch_norm
#   convolutional layers
#   pooling layers (what are they called again?)
#   better way to handle test not using mini-batches
#   implement momentum
#   implement early stopping
#   implement L1 regularization

#   Create a more consistent testing regime:  independent validation set
#   scale weights for cost regularization to accommodate ReLU normalization?
#   separate plots data structure from stats data structure

# data structures for neural network
export NN_parameters, Model_data, Batch_norm_params

# functions to use
export train_nn, test_score, save_theta, accuracy, predictions_vector

using MAT
using PyCall
using Plots
pyplot()  # initialize the backend used by Plots
@pyimport seaborn  # prettier charts
# using ImageView    BIG BUG HERE--SEGFAULT--REPORTED

const l_relu_neg = .01  

"""
struct NN_parameters holds model parameters learned by training and model metadata
"""
mutable struct NN_parameters  
    theta::Array{Array{Float64,2},1}
    bias::Array{Array{Float64,1},1}
    delta_w::Array{Array{Float64,2},1}  
    delta_b::Array{Array{Float64,1},1}  
    theta_dims::Array{Tuple{Int64, Int64},1}
    output_layer::Int64
    layer_units::Array{Int64,1}

    # empty constructor
    NN_parameters() = new(                  
        Array{Array{Float64,2},1}(0),    # theta::Array{Array{Float64,2}}
        Array{Array{Float64,2},1}(0),    # bias::Array{Array{Float64,1}}
        Array{Array{Float64,2},1}(0),    # delta_w
        Array{Array{Float64,2},1}(0),    # delta_b  
        Array{Tuple{Int, Int},1}(0),      # theta_dims::Array{Array{Int64,2}}
        0,                               # output_layer
        Array{Int64,1}(0)                # layer_units
    )
end


"""
Struct Model_data hold examples and all layer outputs-->
pre-allocate to reduce memory allocations and improve speed
"""
mutable struct Model_data  
    inputs::Array{Float64,2}  # k features by n examples
    targets::Array{Float64,2} # labels for each example
    a::Array{Array{Float64,2},1}
    z::Array{Array{Float64,2},1}
    z_norm::Array{Array{Float64,2},1}  # same size as z
    z_scale::Array{Array{Float64,2},1}  # same size as z, often called "y"
    grad::Array{Array{Float64,2},1}
    epsilon::Array{Array{Float64,2},1}

    Model_data() = new(
        Array{Float64,2}(2,2),          # inputs
        Array{Float64,2}(2,2),          # targets
        Array{Array{Float64,2},1}(0),   # a
        Array{Array{Float64,2},1}(0),   # z
        Array{Array{Float64,2},1}(0),   # z_norm -- only pre-allocate if batch_norm
        Array{Array{Float64,2},1}(0),   # z_scale -- only pre-allocate if batch_norm
        Array{Array{Float64,2},1}(0),   # grad
        Array{Array{Float64,2},1}(0)    # epsilon
    )
end


"""
struct Batch_norm_params holds batch normalization parameters for 
feedfwd calculations, and backprop training.
"""
mutable struct Batch_norm_params  # 
    gam::Array{Array{Float64,1},1}  
    bet::Array{Array{Float64,1},1}  
    delta_gam::Array{Array{Float64,1},1}
    delta_bet::Array{Array{Float64,1},1} 
    delta_z_norm::Array{Array{Float64,2},1}  # same size as z
    delta_z_scale::Array{Array{Float64,2},1}  # same size as z
    delta_z::Array{Array{Float64,2},1}  # same size as z
    mu::Array{Array{Float64,1},1}  # same size as bias = no. of layer units
    stddev::Array{Array{Float64,1},1} #    ditto
    mu_run::Array{Array{Float64,1},1}
    std_run::Array{Array{Float64,1},1}

    Batch_norm_params() = new(
        Array{Array{Float64,1},1}(0),    # gam::Array{Array{Float64,1}}
        Array{Array{Float64,1},1}(0),    # bet::Array{Array{Float64,1}}
        Array{Array{Float64,2},1}(0),    # delta_gam
        Array{Array{Float64,2},1}(0),    # delta_bet
        Array{Array{Float64,2},1}(0),    # delta_z_norm
        Array{Array{Float64,2},1}(0),    # delta_z_scale
        Array{Array{Float64,2},1}(0),    # delta_z
        Array{Array{Float64,1},1}(0),    # mu
        Array{Array{Float64,1},1}(0),    # stddev
        Array{Array{Float64,1},1}(0),    # mu_run
        Array{Array{Float64,1},1}(0)     # std_run
    )
end


"""
Layer template to contain metadata for a single layer above input
usage (example with 1 hidden layer and softmax output:
    mylayers = [] # array holds layer dicts for all layers: input=1 to output=n

    mylayer = deepcopy(layer_template) # do this every time to prevent accidental hold over values!
    mylayer["kind"] = "input"
    mylayer["units"] = 40
    push!(mylayers, mylayer) # absolutely push in order from input to output

    mylayer = deepcopy(layer_template) # do this every time to prevent accidental hold over values!
    mylayer["kind"] = "relu"  # note: this is default so could leave as is
    mylayer["units"] = 10
    push!(mylayers, mylayer) # absolutely push in order from input to output

    mylayer = deepcopy(layer_template) # do this every time to prevent accidental hold over values!
    mylayer["kind"] = "softmax"  # note: this is default so could leave as is
    mylayer["units"] = 4
    push!(mylayers, mylayer) # absolutely push in order from input to output    

"""
const layer_template = Dict(
    "kind" => "relu", # valid: "input", "linear", relu", "sigmoid", "softmax", "l_relu", "conv", "pooling"
    "units" => 10,
    "filter_size" => (3,3), # use for conv or pooling
    "conv_filters" => 6,
    "pooling_op" => "max", # valid: "max", "avg"
    "l_relu_neg" => 0.01, # for negative Z values
    )


"""
Method to call train_nn with a single integer as the number of hidden units
in a single hidden layer.
"""
function train_nn(matfname::String, n_iters::Int64, n_hid::Int64; alpha=0.35,
    mb_size=0, lambda=0.015, classify="softmax", normalization::Bool=false,
    units="sigmoid", plots=["Training", "Learning"])

    train_nn(matfname, n_iters, [n_hid]; alpha=alpha,
    mb_size=mb_size, lambda=lambda, classify=classify, normalization=normalization,
    units=units, plots=plots)
end


"""
    function train_nn(matfname::String, n_iters::Int64, n_hid::Array{Int64,1}, alpha=0.35,
        mb_size=0, lambda=0.015, classify="softmax", units="sigmoid", plots=["Training", "Learning"])

    returns theta -- the model weights
    key inputs:
        alpha   ::= learning rate
        lambda  ::= regularization rate
        mb_size ::= mini-batch size

Train sigmoid/softmax neural networks up to 11 layers.  Detects
number of output labels from data. Detects number of features from data for output units. 
Enables any size mini-batch that divides evenly into number of examples.  Plots 
by any choice of "Learning", "Cost", per "Training" iteration (epoch) and for "Test" data.
classify may be "softmax" or "sigmoid", which applies only to the output layer. 
Units in other layers may be "sigmoid," "l_relu" or "relu".
"""
function train_nn(matfname::String, n_iters::Int64, n_hid::Array{Int64,1}; alpha=0.35,
    mb_size::Int64=0, lambda::Float64=0.015,  normalization::Bool=false,
    classify::String="softmax", units::String="sigmoid", 
    plots::Array{String,1}=["Training", "Learning"])

    ################################################################################
    #   This is a front-end function that verifies all inputs and calls run_training
    ################################################################################

    if n_iters < 0
        error("Input n_iters must be an integer greater than 0")
    end

    if ndims(n_hid) != 1
        error("Input n_hid must be a vector.")
    elseif size(n_hid,1) > 9
        error("n_hid can only contain 1 to 9 integer values for 1 to 9 hidden layers.")
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

    if lambda < 0.0  # set lambda = 0.0 for relu with batch_norm
        warn("Lambda regularization rate must be positive floating point value. Setting to 0.")
        lambda = 0.0
    elseif lambda > 5.0
        warn("Lambda regularization rate set too large. Setting to max of 5.0")
        lambda = 5.0
    end

    if !in(classify, ["softmax", "sigmoid"])
        warn("classify must be \"softmax\" or \"sigmoid\". Setting to default \"softmax\".")
        classify = "softmax"
    end

    if !in(units, ["l_relu", "sigmoid", "relu"])
        warn("units must be \"relu,\" \"l_relu,\" or \"sigmoid\". Setting to default \"sigmoid\".")
    end

    valid_plots = ["Training", "Test", "Learning", "Cost"]
    new_plots = [pl for pl in valid_plots if in(pl, plots)]  
    if sort(new_plots) != sort(plots)
        warn("Plots argument can only include \"Training\", \"Test\", \"Learning\", and \"Cost\".\nProceeding with default [\"Training\", \"Learning\"].")
        plots = ["Training", "Learning"]
    else
        plots = new_plots
    end

    theta, batch_params = run_training(matfname, n_iters, plots,
        n_hid, alpha, mb_size, lambda, classify, normalization, units);
end


function run_training(matfname::String, n_iters::Int64, plots::Array{String,1}, 
    n_hid::Array{Int64,1}, alpha=0.35, mb_size=0, lambda=0.015,  
    classify="softmax", normalization=false, units="sigmoid")

    # start the cpu clock
    tic()

    # seed random number generator.  For runs of identical models the same weight initialization
    # will be used, given the number of parameters to be estimated.
    srand(70653)  # seed int value is meaningless    


    ##########################################################
    #   pre-allocate and initialize variables
    ##########################################################

    # create data containers
    train = Model_data()  # train holds all the data and layer inputs/outputs
    test = Model_data()

    # load training data and test data (if any)
    train.inputs, train.targets, test.inputs, test.targets, normfactors = extract_data(matfname, normalization)

    # set some useful variables
    k,n = size(train.inputs)  # number of features k by no. of examples n
    t = size(train.targets,1)  # number of output units
    dotest = size(test.inputs, 1) > 0  # there is testing data
    batch_norm = contains(units, "relu") ? true : false  # determines data series needed

    # neural net model parameters
    p = NN_parameters()  # p holds all the parameters that will be trained and some metadata
    preallocate_nn_params!(p, n_hid, k, n, t)

    # feedfwd training data
    preallocate_feedfwd!(train, p, n, batch_norm) 

    # feedfwd test data--if inputs are not all zeros
    if dotest
        testn = size(test.inputs,2)
        preallocate_feedfwd!(test, p, testn, batch_norm) 
    end

    # statistics for plots and history data
    plotdef = setup_plots(n_iters, dotest, plots)

    #setup mini-batch
    if mb_size == 0
        mb_size = n  # use 1 (mini-)batch with all of the examples
    elseif mb_size > n
        mb_size = n
    elseif mb_size < 1
        mb_size = n
    elseif mod(n, mb_size) != 0
        error("Mini-batch size $mb_size does not divide evenly into samples $n.")
    end
    n_mb = Int(n / mb_size)  # number of mini-batches 
    batch_norm = n_mb == 1 ? false : batch_norm  # no batch normalization for 1 batch
    alphaovermb = alpha / mb_size  # calc once, use in loop

    if mb_size < n
        # randomize order of all training samples: 
            # labels in training data often in a block, which will make
            # mini-batch train badly because a batch will not contain mix of target labels
        sel_index = randperm(n)
        train.inputs[:] = train.inputs[:, sel_index]
        train.targets[:] = train.targets[:, sel_index]  
    end

    # pre-allocate feedfwd mini-batch inputs and targets
    mb = Model_data()  # mb holds all layer data for mini-batches
    preallocate_minibatch!(mb, p, mb_size, k, n, t, batch_norm)
 
    # batch normalization parameters
    bn = Batch_norm_params()  # do we always need the data structure to run?
    if batch_norm
        preallocate_batchnorm!(bn, mb, p.layer_units)
    end


    #################################################################
    #   define and choose functions to be used in neural net training
    #################################################################

    if units == "sigmoid"
        unit_function! = sigmoid!
        batch_norm = false
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

    if t > 1  # more than one output (unit)
        if classify == "sigmoid"
            classify_function! = sigmoid!
        elseif classify == "softmax"
            classify_function! = softmax!
        else
            error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
        end
    else
        classify_function! = sigmoid!  # for one output label
    end

    # now there's only one cost function.  someday there could be others.
    cost_function = cross_entropy_cost

    # define two functions for alternative weight updates
    weight_update_noreg(theta, delta) = theta .- (alphaovermb .* delta)
    weight_update_reg(theta, delta) = theta .- (alphaovermb .* delta) .- (alphaovermb .* (lambda .* theta))

    # choose the weight update function (keep this test out of the training loop)
    if lambda <= 0.0  
        weight_update! = weight_update_noreg 
    else 
        weight_update! = weight_update_reg
    end   

    # function lists to be passed to feedfwd! and backprop!
    fwd_functions = (batch_norm_fwd!, unit_function!, classify_function!)
    back_functions = (batch_norm_back!, gradient_function!)


    ##########################################################
    #   neural network training loop
    ##########################################################

    for i = 1:n_iters  # loop for "epochs"
        for j = 1:n_mb  # loop for mini-batches

            first_example = (j - 1) * mb_size + 1  # mini-batch subset for the inputs->layer 1
            last_example = first_example + mb_size - 1
            
            mb.a[1][:] = train.inputs[:,first_example:last_example] # m-b input layer activation  
            mb.targets[:] = train.targets[:, first_example:last_example]      

            feedfwd!(p, bn, mb, fwd_functions, batch_norm)

            backprop!(p, bn, mb, back_functions, batch_norm)

            # update weights, bias, and batch_norm parameters (if used)
            @fastmath for hl = 2:p.output_layer              
                if batch_norm
                    bn.gam[hl][:] -= alphaovermb .* bn.delta_gam[hl] 
                    bn.bet[hl][:] -= alphaovermb .* bn.delta_bet[hl]
                else
                    p.bias[hl][:] -= alphaovermb .* p.delta_b[hl]  
                end
                p.theta[hl][:] = weight_update!(p.theta[hl], p.delta_w[hl])
            end
        end

        gather_stats!(i, plotdef, mb, test, p, bn, cost_function, fwd_functions, mb_size, testn, 
            lambda, batch_norm)  # ??? TODO train or mb???    
    end

    toc()  # print cpu time since tic()
    
    #####################################################################
    # output and plot training statistics
    #####################################################################
    
    feedfwd!(p, bn, train, fwd_functions, batch_norm, istrain=false)  # output for entire training set
    println("Fraction correct labels predicted training: ", accuracy(train.targets, train.a[p.output_layer],n_iters))
    println("Final cost training: ", cost_function(train.targets, train.a[p.output_layer], n,
                    p.theta, lambda, p.output_layer))

    # output test statistics
    if dotest     
        feedfwd!(p, bn, test, fwd_functions, batch_norm, istrain=false)
        println("Fraction correct labels predicted test: ", accuracy(test.targets, test.a[p.output_layer],n_iters))
        println("Final cost test: ", cost_function(test.targets, test.a[p.output_layer], testn, 
            p.theta, lambda, p.output_layer))
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

    return p, bn;  # trained model parameters

end  # function run_training


function extract_data(matfname::String, normalization)
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

    normfactors = (0.0, 1.0) # x_mu, x_std
    if normalization
        # normalize training data
        x_mu = mean(inputs)
        x_std = std(inputs)
        inputs = (inputs .- x_mu) ./ x_std
        #normalize test data
        if in("test", keys(df))
            test_inputs = (test_inputs .- x_mu) ./ x_std
        end
        normfactors = (x_mu, x_std)
    end
    # normalize test data
    return inputs, targets, test_inputs, test_targets, normfactors
end


function preallocate_feedfwd!(dat, p, n, batch_norm)   
    dat.a = [dat.inputs]
    dat.z = [zeros(2,2)] # not used for input layer
    for i = 2:p.output_layer-1  # hidden layers
        push!(dat.z, zeros(size(p.theta[i] * dat.a[i-1])))  # z2 and up...  ...output layer set after loop
        push!(dat.a, zeros(size(dat.z[i])))  #  and up...  ...output layer set after loop
    end
    push!(dat.z, zeros(size(p.theta[p.output_layer],1),n))  
    push!(dat.a, zeros(size(p.theta[p.output_layer],1),n))  

    if batch_norm
        dat.z_norm = deepcopy(dat.z)  
        dat.z_scale = deepcopy(dat.z)  # same size as z, often called "y"
    end
end


function preallocate_nn_params!(p, n_hid, k, n, t)
    # initialize and pre-allocate data structures to hold neural net training data
    # theta = weight matrices for all calculated layers (e.g., not the input layer)
    # bias = bias term used for every layer but input
    # k = no. of features in input layer
    # n = number of examples in input layer (and throughout the network)
    # t = number of features in the targets--the output layer

    # theta dimensions for each layer of the neural network 
    #    Follows the convention that rows = outputs of the current layer activation
    #    and columns are the inputs from the layer below

    # layers
    p.output_layer = 2 + size(n_hid, 1) # input layer is 1, output layer is highest value
    p.layer_units = [k, n_hid..., t]

    # set dimensions of the linear weights for each layer
    push!(p.theta_dims, (k, 1)) # weight dimensions for the input layer -- if using array, must splat as arg
    for l = 2:p.output_layer-1  # l refers to nn layer so this includes only hidden layers
        push!(p.theta_dims, (n_hid[l-1], p.theta_dims[l-1][1])) # rows = hidden, cols = lower layer outputs
    end
    push!(p.theta_dims, (t, n_hid[end]))  # weight dims for output layer: rows = output classes

    # initialize the linear weights
    p.theta = [zeros(2,2)] # layer 1 not used
    # uniform distribution in (-.5, .5):  no real theoretical justification and less than best results
        # interval = 0.5
        # for i = 2:p.output_layer
        #     push!(p.theta, rand(p.theta_dims[i]) .* (2.0 * interval) .- interval)
        # end
    # Xavier initialization--current best practice for relu
    for l = 2:p.output_layer
        push!(p.theta, randn(p.theta_dims[l]) .* sqrt(2.0/p.theta_dims[l][2])) # sqrt of no. of input units
    end

    # bias initialization: random non-zero initialization performs worse
    p.bias = [zeros(size(th, 1)) for th in p.theta]  # initialize biases to zero
    # p.bias = [rand(size(th,1)) .* (2. * interval) .- interval for th in p.theta] # initialize (-.5, .5)
    # p.bias = [randn(size(th,1)) .* sqrt(2.0/size(th,2)) for th in p.theta]

    # structure of gradient matches theta
    p.delta_w = deepcopy(p.theta)  
    p.delta_b = deepcopy(p.bias)

end


function preallocate_minibatch!(mb, p, mb_size, k, n, t, batch_norm)
    mb.inputs = zeros(k, mb_size)  
    mb.targets = zeros(t, mb_size)  
    preallocate_feedfwd!(mb, p, mb_size, batch_norm)   
    mb.epsilon = deepcopy(mb.a)  # looks like activations of each unit above input layer
    mb.grad = deepcopy(mb.z)
    if batch_norm
        mb.z_norm = deepcopy(mb.z)
        mb.z_scale = deepcopy(mb.z)
    end    
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
    bn.delta_z = deepcopy(mb.z_norm)
    bn.delta_z_norm = deepcopy(mb.z_norm)
    bn.mu = [zeros(i) for i in layer_units]  # same size as bias = no. of layer units
    bn.mu_run = [zeros(i) for i in layer_units]
    bn.stddev = [zeros(i) for i in layer_units] 
    bn.std_run = [zeros(i) for i in layer_units]
end


"""
function feedfwd!(p, bn, dat, fwd_functions, batch_norm; istrain) 
    modifies a, a_wb, z in place to reduce memory allocations
    send it all of the data or a mini-batch

    feed forward from inputs to output layer predictions
"""
function feedfwd!(p, bn, dat, fwd_functions, batch_norm; istrain=true)

    (batch_norm_fwd!, unit_function!, classify_function!) = fwd_functions

    @fastmath for hl = 2:p.output_layer-1  # hidden layers

        if batch_norm
            dat.z[hl][:] = p.theta[hl] * dat.a[hl-1]  # linear with no bias
            batch_norm_fwd!(bn, dat, hl, istrain)
            unit_function!(dat.z_scale[hl],dat.a[hl]) # non-linear function
        else  
            dat.z[hl][:] = p.theta[hl] * dat.a[hl-1] .+ p.bias[hl]  # linear with bias
            unit_function!(dat.z[hl],dat.a[hl])
        end

    end

    @fastmath dat.z[p.output_layer][:] = (p.theta[p.output_layer] * dat.a[p.output_layer-1] 
        .+ p.bias[p.output_layer])  # TODO use bias in the output layer with no batch norm?

    classify_function!(dat.z[p.output_layer], dat.a[p.output_layer])  # a = predictions

end


"""
function backprop!(p, dat, back_functions, batch_norm) 
    Argument p.delta_w holds the computed gradients for weights, delta_b for bias
    Modifies p.epsilon, p.delta_w, p.delta_b in place--caller uses p.delta_w, p.delta_b
    Use for training iterations 
    Send it all of the data or a mini-batch
    Intermediate storage of p.a, p.a_wb, p.z, p.epsilon, p.delta_w, p.delta_b reduces memory allocations
"""
function backprop!(p, bn, dat, back_functions, batch_norm)

    (batch_norm_back!, gradient_function!) = back_functions

    dat.epsilon[p.output_layer][:] = dat.a[p.output_layer] .- dat.targets 
    @fastmath p.delta_w[p.output_layer][:] = dat.epsilon[p.output_layer] * dat.a[p.output_layer-1]'
    @fastmath p.delta_b[p.output_layer][:] = sum(dat.epsilon[p.output_layer],2)

    @fastmath for hl = (p.output_layer - 1):-1:2  # for hidden layers 
        if batch_norm
            gradient_function!(dat.z_scale[hl], dat.grad[hl])  
            dat.epsilon[hl][:] = p.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl]  
            batch_norm_back!(p, dat, bn, hl)
            p.delta_w[hl][:] = bn.delta_z[hl] * dat.a[hl-1]'
        else
            gradient_function!(dat.z[hl], dat.grad[hl])  
            dat.epsilon[hl][:] = p.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl]  
            p.delta_w[hl][:] = dat.epsilon[hl] * dat.a[hl-1]'
            p.delta_b[hl][:] = sum(dat.epsilon[hl],2)  #  times a column of 1's = sum(row)
        end

    end

end


function cross_entropy_cost(targets, predictions, n, theta, lambda, output_layer)
    # n is count of all samples in data set--use with regularization term
    # mb_size is count of all samples used in training batch--use with cost
    # these may be equal
    cost = (-1.0 ./ n) .* sum(targets .* log.(predictions) .+ 
        (1.0 .- targets) .* log.(1.0 .- predictions))
    @fastmath if lambda > 0.0  # don't regularize relu with batch normalization->set lambda=0.0
        # regterm = lambda/(2.0 * n) .* sum(theta[output_layer][:, 2:end] .* theta[output_layer][:, 2:end]) 
        regterm = lambda/(2.0 * n) .* sum([sum(th .* th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end


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
    for j = 1:size(z,2)  # pick a column
        for i = 1:size(z,1)  # down each column for speed
            @.  a[i,j] = z[i,j] >= 0.0 ? z[i,j] : l_relu_neg * z[i,j]
        end
    end
end


function relu!(z::Array{Float64,2}, a::Array{Float64,2}) # leaky relu
    a[:] = max.(z, 0.0)
end


function batch_norm_fwd!(bn, dat, hl, istrain=true)
    k,mb = size(dat.z[hl])
    if istrain
        variance = zeros(k)
        bn.mu[hl][:] = mean(dat.z[hl], 2)          # use in backprop
        variance[:] = 1.0/mb .* sum((dat.z[hl] .- bn.mu[hl]).^2.0, 2)
        bn.stddev[hl][:] = sqrt.(variance .+ 1e-8)      # use in backprop
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


function batch_norm_back!(p, dat, bn, hl)
    d,mb = size(dat.epsilon[hl])  
    bn.delta_bet[hl][:] = sum(dat.epsilon[hl], 2)  
    bn.delta_gam[hl][:] = sum(dat.epsilon[hl] .* dat.z_norm[hl], 2)  
    bn.delta_z_norm[hl][:] = bn.gam[hl] .* dat.epsilon[hl]  
    bn.delta_z[hl][:] = (  
        (1.0 / mb) .* (1.0 ./ bn.stddev[hl]) .* (
            mb .* bn.delta_z_norm[hl] .- sum(bn.delta_z_norm[hl],2) .- 
            dat.z_norm[hl] .* sum(bn.delta_z_norm[hl] .* dat.z_norm[hl], 2)
            )  
        )
end


function softmax!(z::Array{Float64,2}, a::Array{Float64,2})  
    expf = similar(a)
    f = similar(a)
    f[:] = z .- maximum(z,1)  
    expf[:] = exp.(f)  # this gets called within a loop and exp() is expensive
    a[:] = @fastmath expf ./ sum(expf, 1)  
end


# two methods for gradient of linear layer units:  without bias and with
function affine_grad!(currlayerdiff, prevlayeract)  # no bias
    return currlayerdiff * prevlayeract'
end


function affine_grad!(currlayerdiff, prevlayeract, dobias::Bool)
    return currlayerdiff * prevlayeract', sum(currlayerdiff, 2)
end


function sigmoid_gradient!(z::Array{Float64,2}, grad::Array{Float64,2})
    sigmoid!(z, grad)
    grad[:] = grad .* (1.0 .- grad)
end


function l_relu_gradient!(la::Array{Float64,2}, grad::Array{Float64,2})
    for j = 1:size(la, 2)  # calculate down a column for speed
        for i = 1:size(la, 1)
            grad[i,j] = la[i,j] > 0.0 ? 1.0 : l_relu_neg
        end
    end
end


function relu_gradient!(la::Array{Float64,2}, grad::Array{Float64,2})
    for j = 1:size(la, 2)  # calculate down a column for speed
        for i = 1:size(la, 1)
            grad[i,j] = la[i,j] > 0.0 ? 1.0 : 0.0
        end
    end
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
Function setup_plots(n_iters::Int64, dotest::Bool, plots::Array{String,1})

Creates data structure to hold everything needed to plot progress of
neural net training by iteration.

A plotdef is a dict containing:

    "plot_switch"=>plot_switch: Dict of bools for each type of results to be plotted.
        Currently used are: "Training", "Test", "Learning".  This determines what 
        data will be collected during training iterations and what data series will be
        plotted.
    "plot_labels"=>plot_labels: array of strings provides the labels to be used in the
        plot legend. 
    "cost_history"=>cost_history: an array of calculated cost at each iteration 
        with iterations as rows and result types as columns.  
        Results can be Training or Test. 
    "fracright_history"=>fracright_history: an array of percentage of correct classification
        at each iteration with iterations as rows and result types as columns. This plots
        a so-called learning curve.  Very interesting indeed. 
        Results can be Training or Test.  
    "col_train"=>col_train: column of the arrays above to be used for Training results
    "col_test"=>col_test: column of the arrays above to be used for Test results

"""
function setup_plots(n_iters::Int64, dotest::Bool, plots::Array{String,1})
    # set up cost_history to track 1 or 2 data series for plots
    # lots of indirection here:  someday might add "validation"
    if size(plots,1) > 4
        warn("Only 4 plot requests permitted. Proceeding with up to 4.")
    end

    valid_plots = ["Training", "Test", "Learning", "Cost"]
    plot_switch = Dict(pl => in(pl, plots) for pl in valid_plots)

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
        cost_history = zeros(n_iters, size(plot_labels,2))
        plotdef["cost_history"] = cost_history
    end
    if plot_switch["Learning"]
        fracright_history = zeros(n_iters, size(plot_labels,2))
        plotdef["fracright_history"] = fracright_history
    end
 
    # set column in cost_history for each data series
    col_train = plot_switch["Training"] ? 1 : 0
    col_test = plot_switch["Test"] ? col_train + 1 : 0

    plotdef["col_train"] = col_train
    plotdef["col_test"] = col_test

    return plotdef
end


function gather_stats!(i, plotdef, mb, test, p, bn, cost_function, fwd_functions, train_n, testn, 
    lambda, batch_norm)  

    if plotdef["plot_switch"]["Training"]
        if plotdef["plot_switch"]["Cost"]
            plotdef["cost_history"][i, plotdef["col_train"]] = cost_function(mb.targets, 
                mb.a[p.output_layer], train_n, p.theta, lambda, p.output_layer)
        end
        if plotdef["plot_switch"]["Learning"]
            plotdef["fracright_history"][i, plotdef["col_train"]] = accuracy(
                mb.targets, mb.a[p.output_layer],i)
        end
    end
    
    if plotdef["plot_switch"]["Test"]
        if plotdef["plot_switch"]["Cost"]
            feedfwd!(p, bn, test, fwd_functions, batch_norm, istrain=false)  
            plotdef["cost_history"][i, plotdef["col_test"]] = cost_function(test.targets, 
                test.a[p.output_layer], testn, p.theta, lambda, p.output_layer)
        end
        if plotdef["plot_switch"]["Learning"]
            # printdims(Dict("test.a"=>test.a, "test.z"=>test.z))
            feedfwd!(p, bn, test, fwd_functions, batch_norm, istrain=false)  
            plotdef["fracright_history"][i, plotdef["col_test"]] = accuracy(test.targets, 
                test.a[p.output_layer],i)          
        end
    end     
end  


function accuracy(targets, preds,i)
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
        choices = [j >= 0.5 ? 1.0 : 0.0 for j in predictions]
        fracright = mean(convert(Array{Int},choices .== targets))
    end
    return fracright
end


"""

    function save_theta(theta, mat_fname)

Save the weights, or theta, trained by the neural network as a matlab file.
Can be used to run the model on prediction data or to evaluate other 
test data results (cost and accuracy).
"""
function save_theta(theta, mat_fname)
    # check if output file exists and ask permission to overwrite
    if isfile(mat_fname)
        print("Output file $mat_fname exists. OK to overwrite? ")
        resp = readline()
        if contains(lowercase(resp), "y")
            rm(mat_fname)
        else
            error("File exists. Replied no to overwrite. Quitting.")
        end
    end

    # write the matlab formatted file (based on hdf5)
    outfile = matopen(mat_fname, "w")
    write(outfile, "theta", theta)
    close(outfile) 
end


"""

    function test_score(theta_fname, data_fname, lambda = 0.015, 
        classify="softmax")

Calculate the test accuracy score and cost for a dataset containing test or validation
data.  This data must contain outcome labels, e.g.--y.
"""
function test_score(theta_fname, data_fname, lambda = 0.015, classify="softmax")
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
    println("Final cost test: ", cost_function(targets, predictions, n, theta, lambda, output_layer))

    return score
end


"""

    function predictions_vector(theta_fname, data_fname, lambda = 0.015, 
        classify="softmax")

    returns vector of all predictions

Return predicted values given inputs and theta.  Not used by training.
Use when theta is already trained and saved to make predictions
for input operational data to use your model. Resolves sigmoid or softmax outputs
to (zero, one) values for each output unit.

Simply does feedforward, but does some data staging first.
"""
function predictions_vector(theta_fname, data_fname, lambda = 0.015, classify="softmax")
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
    k,n = size(inputs) # number of features k by no. of examples n
    output_layer = size(theta,1) 
    t = size(theta[output_layer],1)  # number of output units or 

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
 
    a_test,  z_test = preallocate_feedfwd(inputs, p, n) 
    predictions = feedfwd!(p, bn, dat, fwd_functions, batch_norm, istrain=false) 
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
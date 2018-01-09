

# Includes the following functions to run directly:
#    train_nn() -- train sigmoid/softmax neural networks for up to 5 layers
#    test_score() -- cost and accuracy given test data and saved theta
#    save_theta() -- save theta, which is returned by train_nn
#    predictions_vector() -- predictions given x and saved theta


#DONE


#TODO
#   fix cost calculation: results in NaN with relu and batch_norm
#   clean up pre-allocation code
#   separate the backprop for normalizing layer and scaling layer
#   what is the divisor for lambda in the regterm of cost????
#   better way to handle test not using mini-batches
#   modify normalized leaky relu: add a linear transform
#       to the normalized result gamma*z + beta with rho and beta being trained for 
#       each unit
#   implement proper prediction when using batch normalizing: pick suitable mean and var to
#       use for a single example or for the test set
#   implement momentum
#   implement early stopping
#   implement L1 regularization
#   implement dropout
#   split stats from the plotdef
#   Create a more consistent testing regime:  independent validation set
#   scale weights for cost regularization to accommodate ReLU normalization?
#   separate plots data structure from stats data structure


using MAT
using PyCall
using Plots
pyplot()  # initialize the backend used by Plots
@pyimport seaborn  # prettier charts
# using ImageView    BIG BUG HERE--SEGFAULT--REPORTED

"holds model parameters learned by training-->pre-allocate and initialize"
mutable struct NN_parameters  
    theta::Array{Array{Float64,2},1}
    theta_dims::Array{Array{Int64,1},1}
    bias::Array{Array{Float64,1},1}
    delta_w::Array{Array{Float64,2},1}  
    delta_b::Array{Array{Float64,1},1}  
    output_layer::Int64

    # empty constructor
    NN_parameters() = new(                  
        Array{Array{Float64,2},1}(0),    # theta::Array{Array{Float64,2}}
        Array{Array{Int64,2},1}(0),      # theta_dims::Array{Array{Int64,2}}
        Array{Array{Float64,2},1}(0),    # bias::Array{Array{Float64,1}}
        Array{Array{Float64,2},1}(0),    # delta_w
        Array{Array{Float64,2},1}(0),    # delta_b  
        0                                # output_layer
    )
end


"hold examples and all layer outputs-->pre-allocate to reduce memory allocations and improve speed"
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

"batch normalization parameters and data for backprop"
mutable struct Batch_norm_back_data  # 
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

    Batch_norm_back_data() = new(
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
Method to call train_nn with a single integer as the number of hidden units
in a single hidden layer.
"""
function train_nn(matfname::String, n_iters::Int64, n_hid::Int64; alpha=0.35,
    mb_size=0, lambda=0.015, scale_reg::Bool=false, classify="softmax", 
    units="sigmoid", plots=["Training", "Learning"])

    train_nn(matfname, n_iters, [n_hid]; alpha=alpha,
    mb_size=mb_size, lambda=lambda, scale_reg=scale_reg, 
    classify=classify, units=units, plots=plots)
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
Units in other layers may be "sigmoid" or "relu".
"""
function train_nn(matfname::String, n_iters::Int64, n_hid::Array{Int64,1}; alpha=0.35,
    mb_size::Int64=0, lambda::Float64=0.015, scale_reg::Bool=false, 
    classify::String="softmax", units::String="sigmoid", 
    plots::Array{String,1}=["Training", "Learning"])

    #check inputs
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
        error("Input mb_size must be an integer greater than 0")
    end

    if lambda < 0.0
        warn("Lambda regularization rate must be positive floating point value. Setting to 0.")
        lambda = 0.0
    elseif lambda > 5.0
        warn("Lambda regularization rate set too large. Setting to defaut 5.0")
        lambda = 5.0
    end

    if !in(classify, ["softmax", "sigmoid"])
        warn("classify must be \"softmax\" or \"sigmoid\". Setting to default \"softmax\".")
        classify = "softmax"
    end

    if !in(units, ["relu", "sigmoid"])
        warn("units must be \"relu\" or \"sigmoid\". Setting to default \"sigmoid\".")
    end

    valid_plots = ["Training", "Test", "Learning", "Cost"]
    new_plots = [pl for pl in valid_plots if in(pl, plots)]  
    if sort(new_plots) != sort(plots)
        warn("Plots argument can only include \"Training\", \"Test\", \"Learning\", and \"Cost\".\nProceeding with default [\"Training\", \"Learning\"].")
        new_plots = ["Training", "Learning"]
    end

    # create data containers
    train = Model_data()  # train holds all the data and layer inputs/outputs
    test = Model_data()

    # read file and extract data
    train.inputs, train.targets, test.inputs, test.targets = extract_data(matfname)

    # create plot definition
    dotest = size(test.inputs, 1) > 0  # it's true there is test data
    plotdef = setup_plots(n_iters, dotest, new_plots)

    theta = run_training(train, test, n_iters, plotdef,
        n_hid, alpha, mb_size, lambda, scale_reg, classify, units);


end


function extract_data(matfname::String)
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
    return inputs, targets, test_inputs, test_targets
end


function run_training(train::Model_data, test::Model_data, n_iters::Int64, 
    plotdef, n_hid::Array{Int64,1}, alpha=0.35, mb_size=0, lambda=0.015, scale_reg=false, 
    classify="softmax", units="sigmoid")

    ###############################################################
    # do some setup of helpful variables
    ###############################################################

    # start the cpu clock
    tic()

    # seed random number generator.  For runs of identical models the same weight initialization
    # will be used, given the number of parameters to be estimated.
    srand(70653)  # seed int value is meaningless    

    # set some useful variables
    k,n = size(train.inputs)  # number of features k by no. of examples n
    t = size(train.targets,1)  # number of output units

    # layers
    n_hid_layers = size(n_hid, 1)
    output_layer = 2 + n_hid_layers # input layer is 1, output layer is highest value
    layer_units = [k, n_hid..., t]

    # miscellaneous
    dotest = size(test.inputs, 1) > 0
        
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
    alphaovermb = alpha / mb_size  # calc once, use in loop

    if mb_size < n
        # randomize order of all training samples: 
            # labels in training data often in a block, which will make
            # mini-batch train badly because a batch will not contain mix of labels
        sel_index = randperm(n)
        train.inputs[:] = train.inputs[:, sel_index]
        train.targets[:] = train.targets[:, sel_index]  
    end



    #################################################################
    #   define and choose functions to be used in neural net training
    #################################################################

    if units == "sigmoid"
        unit_function! = sigmoid!
        batch_norm = false
    elseif units == "relu"
        unit_function! = l_relu!
        batch_norm = true
    end

    if unit_function! == sigmoid!
        gradient_function! = sigmoid_gradient!
    elseif unit_function! == l_relu!
        gradient_function! = relu_gradient!
    end

    # TODO test theta itself not theta_dims
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

    # define three functions for alternative weight updates
    weight_update_noreg(theta, delta, hl) = theta .- (alphaovermb .* delta)
    weight_update_scale_reg(theta, delta, hl) = (theta .- ((alphaovermb .* delta) .+
        (lambda / layer_units[hl] .* theta)))
    weight_update_reg(theta, delta, hl) = theta .- ((alphaovermb .* delta) .+ (lambda .* theta))

    # choose the weight update function (keep this test out of the training loop)
    if lambda <= 0.0  
        weight_update! = weight_update_noreg
    elseif scale_reg
        weight_update! = weight_update_scale_reg  
    else 
        weight_update! = weight_update_reg
    end   

    # function lists to be passed to feedfwd! and backprop!
    fwd_functions = (batch_norm_fwd!, unit_function!, classify_function!)
    back_functions = (batch_norm_back!, gradient_function!)



    ##########################################################
    #   pre-allocate and initialize variables
    ##########################################################

    # initialize and pre-allocate data structures to hold neural net training data
    # theta = weight matrices for all calculated layers (e.g., not the input layer)
    # bias = bias term used for every layer but input

    # theta dimensions for each layer of the neural network 
    #    this follows the convention that the outputs of the current layer activation
    #    are rows of theta and the inputs from the layer below are columns
    p = NN_parameters()  # p holds all the parameters that will be trained and some metadata
    push!(p.theta_dims, [k, 1]) # weight dimensions for the input layer
    for i = 2:output_layer-1  # hidden layers
        push!(p.theta_dims, [n_hid[i-1], p.theta_dims[i-1][1]]) 
    end
    push!(p.theta_dims, [t, n_hid[end]])  # weight dimensions for the output layer
    p.output_layer = output_layer

    p.theta = [zeros(2,2)] # layer 1 not used
    interval = 0.5 # random weights will be generated in [-interval, interval]
    for i = 2:output_layer
        push!(p.theta, rand(p.theta_dims[i]...) .* (2.0 * interval) .- interval)
    end
    p.bias = [zeros(size(th, 1)) for th in p.theta]
    p.delta_w = deepcopy(p.theta)  # structure of gradient matches theta
    p.delta_b = deepcopy(p.bias)

    # pre-allocate mini-batch inputs and targets
    mb = Model_data()  # mb holds all layer data for mini-batches
    mb.inputs = zeros(k, mb_size)  
    mb.targets = zeros(t, mb_size)  
    mb.a, mb.z = preallocate_feedfwd(mb.inputs, p, mb_size)
    mb.epsilon = deepcopy(mb.a)  # looks like activations of each unit above input layer
    mb.grad = deepcopy(mb.z)
    if batch_norm
        mb.z_norm = deepcopy(mb.z)
        mb.z_scale = deepcopy(mb.z)
    end

    # pre-allocate full training data by layer 
    train.a, train.z = preallocate_feedfwd(train.inputs, p, n)  
    if batch_norm
        train.z_norm = deepcopy(train.z)  
        train.z_scale = deepcopy(train.z)  # same size as z, often called "y"
    end
  
    if dotest
        testn = size(test.inputs,2)
        test.a, test.z = preallocate_feedfwd(test.inputs, p, testn) 
        # test.mb_a, test.mb_z = test.a, test.z
        if batch_norm  # feedfwd of test data using learned batchnorm parameters
            test.z_norm = deepcopy(test.z)
            test.z_scale = deepcopy(test.z)  # same size as mb_z, often called "y"
        end
    end

    # initialize batch normalization parameters gamma and beta
    # vector at each layer corresponding to no. of inputs from preceding layer, roughly "features"
    # gamma = scaling factor for normalization standard deviation
    # beta = bias, or new mean instead of zero
    # should batch normalize for relu, can do for other unit functions
    bn = Batch_norm_back_data()  
    if batch_norm
        bn.gam = [ones(i) for i in layer_units]  # gamma is a builtin function
        bn.bet = [zeros(i) for i in layer_units] # beta is a builtin function
        bn.delta_gam = [zeros(i) for i in layer_units]
        bn.delta_bet = [zeros(i) for i in layer_units]
        bn.delta_z = deepcopy(mb.z_norm)
        bn.delta_z_norm = deepcopy(mb.z_norm)
        bn.mu = [zeros(i) for i in layer_units]  # same size as bias = no. of layer units
        bn.mu_run = [zeros(i) for i in layer_units]
        bn.stddev = [zeros(i) for i in layer_units] #    ditto
        bn.std_run = [zeros(i) for i in layer_units]
    end




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


            # println("***************** a 1 ****************")
            # display(sum(mb.a[1],1))

            backprop!(p, bn, mb, back_functions, batch_norm)

            # println("*************** gam iter $i batch $j ****************")
            # display(bn.gam[2])
            # println("*************** epsilon iter $i batch $j ****************")
            # display(mb.epsilon[2])
            # println("*************** z_norm iter $i batch $j ****************")
            # display(mb.z_norm[2])

            # update weights and bias 
            @fastmath for hl = 2:output_layer              
                if batch_norm
                    bn.gam[hl][:] -= alphaovermb .* bn.delta_gam[hl] 
                    bn.bet[hl][:] -= alphaovermb .* bn.delta_bet[hl]
                else
                    p.bias[hl][:] -= alphaovermb .* p.delta_b[hl]  
                end
                p.theta[hl][:] = weight_update!(p.theta[hl], p.delta_w[hl], hl)
            end
        end


        # println("a:      ", mb.a[p.output_layer][:,1])
        # println("target: ", mb.targets[:,1])
        
        # println(mb.a[p.output_layer][:,50])
        # println(mb.targets[:,50])

        gather_stats!(i, plotdef, mb, test, p, bn, cost_function, fwd_functions, n, 
            lambda, batch_norm)  # ??? TODO train or mb???
        
    end

    toc()  # print cpu time since tic()
    
    #####################################################################
    # output and plot training statistics
    #####################################################################
    
    feedfwd!(p, bn, train, fwd_functions, batch_norm, istrain=false)  
    println("Fraction correct labels predicted training: ", accuracy(train.targets, train.a[p.output_layer]))
    println("Final cost training: ", cost_function(train.targets, train.a[p.output_layer], n,
                    n, p.theta, lambda, output_layer))

    # output improvement of last 10 iterations for test samples
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

    # output test statistics
    if dotest     
        feedfwd!(p, bn, test, fwd_functions, batch_norm, istrain=false)
        println("Fraction correct labels predicted test: ", accuracy(test.targets, test.a[p.output_layer]))
        println("Final cost test: ", cost_function(test.targets, test.a[p.output_layer], testn, testn, 
            p.theta, lambda, output_layer))
    end

    # plot the progress of training cost and/or learning
    plot_output(plotdef)

    return p, bn;

end  # function run_training


function preallocate_feedfwd(inputs, p, n)
    a = [inputs]
    z = [zeros(2,2)] # not used for input layer
    for i = 2:p.output_layer-1  # hidden layers
        push!(z, zeros(size(p.theta[i] * a[i-1])))  # z2 and up...  ...output layer set after loop
        push!(a, zeros(size(z[i])))  #  and up...  ...output layer set after loop
    end
    push!(z, zeros(size(p.theta[p.output_layer],1),n))  
    push!(a, zeros(size(p.theta[p.output_layer],1),n))  

    return a, z  
end


function feedfwd!(p, bn, dat, fwd_functions, batch_norm; istrain=true)
# function feedfwd!(theta, bias, output_layer, unit_function!, classify_function!, a, z) 
    # modifies a, a_wb, z in place to reduce memory allocations
    # send it all of the data or a mini-batch

    # feed forward from inputs to output layer predictions
    # if batch normalized relu, we need to ignore bias

    (batch_norm_fwd!, unit_function!, classify_function!) = fwd_functions

    @fastmath for hl = 2:p.output_layer-1  # hidden layers

        if batch_norm
            dat.z[hl][:] = p.theta[hl] * dat.a[hl-1]  # no bias
            batch_norm_fwd!(bn, dat, hl, istrain)
            unit_function!(dat.z_scale[hl],dat.a[hl])
        else  
            dat.z[hl][:] = p.theta[hl] * dat.a[hl-1] .+ p.bias[hl]
            unit_function!(dat.z[hl],dat.a[hl])
        end

        # println("*********  theta   ************")
        # display(p.theta[hl])
        # println("*********  z   ************")
        # display(dat.z[hl])
        # println("*********  z_norm   ************")
        # display(dat.z_norm[hl])
        # println("*********  z_scale   ************")
        # display(dat.z_scale[hl])
        # println("*********  a   ************")
        # display(dat.a[hl])


    end



    @fastmath dat.z[p.output_layer][:] = (p.theta[p.output_layer] * dat.a[p.output_layer-1] 
        .+ p.bias[p.output_layer])  # TODO use bias in the output layer with no batch norm?

    # println("before: ", dat.a[p.output_layer][1,5])

    classify_function!(dat.z[p.output_layer], dat.a[p.output_layer])

    # println("*********  theta output_layer  ************")
    # display(p.theta[p.output_layer])
    # println("*********  z output_layer  ************")
    # display(dat.z[p.output_layer])
    # println("*********  a output_layer  ************")
    # display(dat.a[p.output_layer])
    # error("that's all folks...")

    # println("after:  ", dat.a[p.output_layer][1,5])

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

    # println("*************** epsilon output_layer ****************")
    # display(dat.epsilon[p.output_layer])


    @fastmath for hl = (p.output_layer - 1):-1:2  # for hidden layers
        # gradient and backprop for the non-linear function
        gradient_function!(dat.z[hl], dat.grad[hl]) 
        dat.epsilon[hl][:] = p.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl]  
        if batch_norm
            batch_norm_back!(p, dat, bn, hl)
            p.delta_w[hl][:] = bn.delta_z[hl] * dat.a[hl-1]'
        else
            p.delta_w[hl][:] = dat.epsilon[hl] * dat.a[hl-1]'
            p.delta_b[hl][:] = sum(dat.epsilon[hl],2)  #  times a column of 1's = sum(row)
        end

    end

end


function cross_entropy_cost(targets, predictions, n, mb_size, theta, lambda, output_layer)
    # n is count of all samples in data set--use with regularization term
    # mb_size is count of all samples used in training batch--use with cost
    # these may be equal
    cost = (-1.0 ./ mb_size) .* sum(targets .* log.(predictions) .+ 
        (1.0 .- targets) .* log.(1.0 .- predictions))
    @fastmath if lambda > 0.0
        # need to fix hidden layer regularization for normalized ReLU
        # because the weights grow because the activations (z) are so small (mean = 0)
        # regterm = lambda/(2.0 * n) * sum([sum(theta[i][:, 2:end] .* theta[i][:, 2:end]) 
        #     for i in 2:output_layer]) 
        regterm = lambda/(2.0 * n) .* sum(theta[output_layer][:, 2:end] .* theta[output_layer][:, 2:end]) 
        cost = cost + regterm
    end
    return cost
end


function sigmoid!(z::Array{Float64,2}, a::Array{Float64,2}) 
    a[:] = 1.0 ./ (1.0 .+ exp.(-z))
end


function l_relu!(z::Array{Float64,2}, a::Array{Float64,2}) 
    # this is leaky relu
    # a[:] = (z .- mean(z,2)) ./ (std(z,2))  # TODO -- take this out once scaling works
    for j = 1:size(z,2)  # down each column for speed
        for i = 1:size(z,1)
            @. a[i,j] = z[i,j] >= 0.0 ? z[i,j] : .01 * z[i,j]
        end
    end
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
    else
        dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu_run[hl]) ./ bn.std_run[hl]  # normalized: 'aka' xhat or zhat
        dat.z_scale[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: 'aka' y   
    end
end


function batch_norm_back!(p, dat, bn, hl)

    d,mb = size(dat.epsilon[hl])  
    bn.delta_bet[hl][:] = sum(dat.epsilon[hl], 2)  # this gets very big
    bn.delta_gam[hl][:] = sum(dat.epsilon[hl] .* dat.z_norm[hl], 2)  # this gets very big
    bn.delta_z_norm[hl][:] = bn.gam[hl] .* dat.epsilon[hl]  
    bn.delta_z[hl][:] = (  
        (1.0 / mb) .* (1.0 ./ bn.stddev[hl]) .* 
        (
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

# works on a single layer at a time
function sigmoid_gradient!(z::Array{Float64,2}, grad::Array{Float64,2})
    sigmoid!(z, grad)
    grad[:] = grad .* (1.0 .- grad)
end

# works on a single layer at a time. la=layer activation
function relu_gradient!(la::Array{Float64,2}, grad::Array{Float64,2})
    for j = 1:size(la, 2)  # calculate down a column for speed
        for i = 1:size(la, 1)
            grad[i,j] = la[i,j] > 0.0 ? 1.0 : .01
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
    if size(plots,1) > 3
        warn("Only 3 plot requests permitted. Proceeding with up to 3.")
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


function gather_stats!(i, plotdef, mb, test, p, bn, cost_function, fwd_functions, n, 
    lambda, batch_norm)  

    if plotdef["plot_switch"]["Training"]
        if plotdef["plot_switch"]["Cost"]
            plotdef["cost_history"][i, plotdef["col_train"]] = cost_function(mb.targets, 
                mb.a[p.output_layer], n, n, p.theta, lambda, p.output_layer)
        end
        if plotdef["plot_switch"]["Learning"]
            plotdef["fracright_history"][i, plotdef["col_train"]] = accuracy(
                mb.targets, mb.a[p.output_layer])
        end
    end
    
    if plotdef["plot_switch"]["Test"]
        if plotdef["plot_switch"]["Cost"]
            feedfwd!(p, bn, test, fwd_functions, batch_norm, istrain=false)  
            plotdef["cost_history"][i, plotdef["col_test"]] = cost_function(test.targets, 
                test.a[p.output_layer], n, n, p.theta, lambda, p.output_layer)
        end
        if plotdef["plot_switch"]["Learning"]
            # printdims(Dict("test.a"=>test.a, "test.z"=>test.z))
            feedfwd!(p, bn, test, fwd_functions, batch_norm, istrain=false)  
            plotdef["fracright_history"][i, plotdef["col_test"]] = accuracy(test.targets, 
                test.a[p.output_layer])
        end
    end     
end  


function accuracy(targets, predictions)
    if size(targets,1) > 1
        # works for output units sigmoid or softmax
        targetmax = [indmax(targets[:,i]) for i in 1:size(targets,2)]
        predmax = [indmax(predictions[:,i]) for i in 1:size(predictions,2)]
        fracright = mean(convert(Array{Int},targetmax .== predmax))
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
    println("Final cost test: ", cost_function(targets, predictions, n, n, theta, lambda, output_layer))

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


"""
Pass a dim1 x dim2 by 1 column vector holding the image data to display it.
Also pass the dimensions as 2 element vector (default is [28,28]).
"""
function display_mnist_digit(digit_data, digit_dims=[28,28])
    imshow(reshape(digit_data, digit_dims...)'); # transpose because inputs were transposed
    println("Press enter to close image window..."); readline()
    ImageView.closeall()
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
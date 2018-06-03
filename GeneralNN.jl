#DONE



#TODO
#   test the setting of the droplim variable for different number of layers
#   try different versions of ensemble predictions_vector
#   allow dropout to drop from the input layer
#   augment data by perturbing the images
#   don't create plotdef if not plotting
#   try batch norm with minmax normalization
#   cleaning up batch norm is complicated:
#       affects feedfwd, backprop, pre-allocation (several), momentum, adam search for if [!hp.]*do_batch_norm
#       type dispatch on bn:  either a struct or a bool to eliminate if tests all over to see if we batch normalize
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
#   separate plots data structure from stats data structure?



"""
Module GeneralNN:

Includes the following functions to run directly:

- train_nn() -- train sigmoid/softmax neural networks for up to 9 hidden layers
- test_score() -- cost and accuracy given test data and saved theta
- save_theta() -- save theta, which is returned by train_nn
- predictions_vector() -- predictions given x and saved theta
- accuracy() -- calculates accuracy of predictions compared to actual labels
- extract_data() -- extracts data for MNIST from matlab files
- normalize_data() -- normalize via standardization (mean, sd) or minmax

To use, include() the file.  Then enter using GeneralNN to use the module.

These data structures are used to hold parameters and data:

- NN_parameters holds theta, bias, delta_w, delta_b, theta_dims, output_layer, layer_units
- Model_data holds inputs, targets, a, z, z_norm, z_scale, epsilon, gradient_function
- Batch_norm_params holds gam (gamma), bet (beta) for batch normalization and Intermediate
data used for training: delta_gam, delta_bet, delta_z_norm, delta_z, mu, stddev, mu_run, std_run

"""
module GeneralNN


# ----------------------------------------------------------------------------------------

# data structures for neural network
export NN_parameters, Model_data, Batch_norm_params, Hyper_parameters

# functions to use
export train_nn, test_score, save_params, load_params, accuracy, predictions_vector, extract_data, normalize_data

using MAT
using JLD
using PyCall
using Plots
pyplot()  # initialize the backend used by Plots
@pyimport seaborn  # prettier charts
# using ImageView    BIG BUG HERE--SEGFAULT--REPORTED

include("layer_functions.jl")
include("nn_data_structs.jl")

const l_relu_neg = .01  # makes the type constant; value can be changed

# ----------------------------------------------------------------------------------------


"""
function train_nn(matfname::String, epochs::Int64, n_hid::Array{Int64,1}; alpha::Float64=0.35,
    mb_size::Int64=0, lambda::Float64=0.01, classify::String="softmax", norm_mode::String="none",
    opt::String="", opt_params::Array{Float64,1}=[0.9,0.999], units::String="sigmoid", do_batch_norm::Bool=false,
    reg::String="L2", dropout::Bool=false, droplim::Array{Float64,1}=[0.5], plots::Array{String,1}=["Training", "Learning"],
    learn_decay::Array{Float64,1}=[1.0, 1.0])

Train sigmoid/softmax neural networks up to 11 layers.  Detects
number of output labels from data. Detects number of features from data for output units.
Enables any size mini-batch that divides evenly into number of examples.  Plots

    returns:
        NN_parameters  ::= struct that holds all trainable parameters (except...)
        Batch_norm_params ::= struct that holds all batch_norm parameters

    key inputs:
        alpha           ::= learning rate
        lambda          ::= regularization rate
        mb_size         ::= mini-batch size=>integer, use 0 to run 1 batch of all examples,
                            otherwise must be an even divisor of the number of examples
        n_hid           ::= array of Int containing number of units in each hidden layer;
                            make sure to use an array even with 1 hidden layer as in [40];
                            use [0] to indicate no hidden layer (typically for linear regression)
        norm_mode       ::= "standard", "minmax" or false => normalize inputs
        do_batch_norm   ::= true or false => normalize each linear layer outputs
        opt             ::= one of "Momentum", "Adam" or "".  default is blank string "".
        opt_params      ::= parameters used by Momentum or Adam
                           Momentum: one floating point value as [.9] (showing default)
                           Adam: 2 floating point values as [.9, .999] (showing defaults)
                           Note that epsilon is ALWAYS set to 1e-8
                           To accept defaults, don't input this parameter or use []
        classify        ::= "softmax", "sigmoid", or "regression" for only the output layer
        units           ::= "sigmoid", "l_relu", "relu" for non-linear activation of all hidden layers
        plots           ::= determines training results collected and plotted
                            any choice of ["Learning", "Cost", "Training", "Test"];
                            for no plots use [""] or ["none"]
        reg             ::= type of regularization, must be one of "L2","Dropout", ""
        dropout         ::= true to use dropout network or false
        droplim         ::= array of values between 0.5 and 1.0 determines how much dropout for
                            hidden layers and output layer (ex: [0.8] or [0.8,0.9, 1.0]).  A single
                            value will be applied to all layers.  If fewer values than layers, then the
                            last value extends to remaining layers.
        learn_decay     ::= array of 2 float values:  first is > 0.0 and <= 1.0 which is pct. reduction of 
                            learning rate; second is >= 1.0 and <= 10.0 for number of times to 
                            reduce learning decay_rate
                            [1.0, 1.0] signals don't do learning decay


"""
function train_nn(matfname::String, epochs::Int64, n_hid::Array{Int64,1}; alpha::Float64=0.35,
    mb_size::Int64=0, lambda::Float64=0.01, classify::String="softmax", norm_mode::String="none",
    opt::String="", opt_params::Array{Float64,1}=[0.9,0.999], units::String="sigmoid", do_batch_norm::Bool=false,
    reg::String="L2", dropout::Bool=false, droplim::Array{Float64,1}=[0.5], plots::Array{String,1}=["Training", "Learning"],
    learn_decay::Array{Float64,1}=[1.0, 1.0])

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

    if alpha < 0.000001
        warn("Alpha learning rate set too small. Setting to default 0.35")
        alpha = 0.35
    elseif alpha > 3.0
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

    norm_mode = lowercase(norm_mode)
    if !in(norm_mode, ["", "none", "standard", "minmax"])
        warn("Invalid norm mode: $norm_mode, using \"none\".")
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

    opt = lowercase(opt)  # match title case for string argument
    if !in(opt, ["momentum", "adam", ""])
        warn("opt must be \"momentum\" or \"adam\" or \"\" (nothing).  Setting to \"\" (nothing).")
        opt = ""
    elseif in(opt, ["momentum", "adam"])
        if size(opt_params) == (2,)
            if opt_params[1] > 1.0 || opt_params[1] < 0.5
                warn("First opt_params for momentum or adam should be between 0.5 and 0.999. Using default")
                opt_params[1] = 0.9
            end
            if opt_params[2] > 1.0 || opt_params[2] < 0.8
                warn("second opt_params for adam should be between 0.8 and 0.999. Using default")
                opt_params[2] = 0.999
            end
        else
            warn("opt_params must be 2 element array with values between 0.9 and 0.999. Using default")
            opt_params = [0.9, 0.999]
        end
    end

    reg = titlecase(lowercase(reg))
    if !in(reg, ["L2", ""])
        warn("reg must be \"L2\" or \"\" (nothing). Setting to default \"L2\".")
        reg = "L2"
    end

    if dropout
        if !all([(c>=.2 && c<=1.0) for c in droplim])
            error("droplim values must be between 0.2 and 1.0. Quitting.")
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

    if size(learn_decay) != (2,)
        warn("learn_decay must be a vector of 2 numbers. Using no learn_decay")
        learn_decay = [1.0, 1.0]
    elseif !(learn_decay[1] >= 0.0 && learn_decay[1] <= 1.0)
        warn("First value of learn_decay must be >= 0.0 and < 1.0. Using no learn_decay")
        learn_decay = [1.0, 1.0]
    elseif !(learn_decay[2] >= 1.0 && learn_decay[2] < 10.0)
        warn("Second value of learn_decay must be >= 1.0 and <= 10.0. Using no learn_decay")
        learn_decay = [1.0, 1.0]
    end

    valid_plots = ["Training", "Test", "Learning", "Cost", "None", ""]
    plots = titlecase.(lowercase.(plots))  # user input doesn't have use perfect case
    new_plots = [pl for pl in valid_plots if in(pl, plots)] # both valid and input by the user
    if sort(new_plots) != sort(plots) # the plots list included something not in valid_plots
        warn("Plots argument can only include \"Training\", \"Test\", \"Learning\" and \"Cost\" or \"None\" or \"\".
            \nProceeding no plots [\"None\"].")
        plots = ["None"]
    else
        plots = new_plots
    end

    run_training(matfname, epochs, n_hid,
        plots=plots, reg=reg, alpha=alpha, mb_size=mb_size, lambda=lambda,
        opt=opt, opt_params=opt_params, classify=classify, dropout=dropout, droplim=droplim,
        norm_mode=norm_mode, do_batch_norm=do_batch_norm, units=units, learn_decay=learn_decay);
end


function run_training(matfname::String, epochs::Int64, n_hid::Array{Int64,1};
    plots::Array{String,1}=["Training", "Learning"], reg="L2", alpha=0.35,
    mb_size=0, lambda=0.01, opt="", opt_params=[], dropout=false, droplim=[0.5],
    classify="softmax", norm_mode="none", do_batch_norm=false, units="sigmoid",
    learn_decay::Array{Float64,1}=[1.0, 1.0])

    # start the cpu clock
    tic()

    # seed random number generator.  For runs of identical models the same weight initialization
    # will be used, given the number of parameters to be estimated.  Enables better comparisons.
    srand(70653)  # seed int value is meaningless


    ##################################################################################
    #   setup model: data structs, many control parameters, functions,  pre-allocation
    #################################################################################

    # instantiate data containers
    train = Model_data()  # train holds all the data and layer inputs/outputs
    test = Model_data()
    mb = Training_view()  # layer data for mini-batches: as views on training data or arrays
    tp = NN_parameters()  # trained parameters
    bn = Batch_norm_params()  # do we always need the data structure to run?  yes--TODO fix this

    hp = Hyper_parameters()  # hyper_parameters:  sets the defaults!
    # update Hyper_parameters with user inputs--more below
        hp.alpha = alpha
        hp.lambda = lambda
        hp.n_hid = n_hid
        hp.reg = reg
        hp.classify = classify
        hp.dropout = dropout
        hp.droplim = droplim
        hp.epochs = epochs
        hp.mb_size = mb_size
        hp.norm_mode = norm_mode
        hp.opt = opt
        hp.opt_params = opt_params
        hp.learn_decay = learn_decay
        hp.do_batch_norm = do_batch_norm

    # load training data and test data (if any)
    train.inputs, train.targets, test.inputs, test.targets = extract_data(matfname)

    # normalize input data
    if !(norm_mode == "" || lowercase(norm_mode) == "none")
        train.inputs, test.inputs, norm_factors = normalize_data(train.inputs, test.inputs, norm_mode)
        tp.norm_factors = norm_factors   
    end
    # debug
    # println("norm_factors ", typeof(norm_factors))
    # println(norm_factors)

    # set some useful variables
    train.in_k, train.n = size(train.inputs)  # number of features in_k (rows) by no. of examples n (columns)
    train.out_k = size(train.targets,1)  # number of output units
    dotest = size(test.inputs, 1) > 0  # there is testing data -> true, else false

    #  optimization parameters, minibatch, preallocate data storage
    setup_model!(mb, hp, tp, bn, dotest, train, test)

    # choose layer functions and cost function based on inputs
    setup_functions(units, train.out_k, opt, classify)

       # statistics for plots and history data
    plotdef = setup_plots(epochs, dotest, plots)

    ##########################################################
    #   neural network training loop
    ##########################################################

    t = 0  # update counter:  epoch by mini-batch

    # DEBUG
    # println(train.inputs)
    # println(train.targets)

    for ep_i = 1:hp.epochs  # loop for "epochs" with counter epoch i as ep_i

        hp.do_learn_decay && step_lrn_decay!(hp, ep_i)

        for mb_j = 1:hp.n_mb  # loop for mini-batches with counter minibatch j as mb_j

            first_example = (mb_j - 1) * hp.mb_size + 1  # mini-batch subset for the inputs->layer 1
            last_example = first_example + hp.mb_size - 1
            colrng = first_example:last_example

            t += 1  # update counter

            update_training_views!(mb, train, tp, hp, colrng)  # next minibatch

            feedfwd!(tp, bn, mb, hp)  # for all layers

            backprop!(tp, bn, mb, hp, t)  # for all layers

            optimization_function!(tp, hp, t)

            # update weights, bias, and batch_norm parameters
            @fastmath for hl = 2:tp.output_layer            
                tp.theta[hl] .= tp.theta[hl] .- (hp.alphaovermb .* tp.delta_w[hl])
                if hp.reg == "L2"  # subtract regularization term
                    tp.theta[hl] .= tp.theta[hl] .- (hp.alphaovermb .* (hp.lambda .* tp.theta[hl]))
                end
                
                if hp.do_batch_norm  # update batch normalization parameters
                    bn.gam[hl][:] -= hp.alphaovermb .* bn.delta_gam[hl]
                    bn.bet[hl][:] -= hp.alphaovermb .* bn.delta_bet[hl]
                else  # update bias
                    tp.bias[hl] .= tp.bias[hl] .- (hp.alphaovermb .* tp.delta_b[hl])
                end

            end  # layer loop
        end # mini-batch loop

        # stats for all mini-batches of one epoch
        gather_stats!(ep_i, plotdef, train, test, tp, bn, cost_function, train.n, test.n, hp)  

    end # epoch loop

    #####################################################################
    # print and plot training statistics after all epochs
    #####################################################################

    println("Training time: ",toq()," seconds")  # cpu time since tic() =>  toq() returns secs without printing

    feedfwd!(tp, bn, train, hp, istrain=false)  # output for entire training set
    println("Fraction correct labels predicted training: ",
            hp.classify == "regression" ? r_squared(train.targets, train.a[tp.output_layer])
                : accuracy(train.targets, train.a[tp.output_layer],epochs))
    println("Final cost training: ", cost_function(train.targets, train.a[tp.output_layer], train.n,
                    tp.theta, hp, tp.output_layer))

    # output test statistics
    if dotest
        feedfwd!(tp, bn, test, hp, istrain=false)
        println("Fraction correct labels predicted test: ",
                hp.classify == "regression" ? r_squared(test.targets, test.a[tp.output_layer])
                    : accuracy(test.targets, test.a[tp.output_layer],epochs))
        println("Final cost test: ", cost_function(test.targets, test.a[tp.output_layer], test.n,
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

    return train.a[tp.output_layer], test.a[tp.output_layer], tp, bn, hp;  
    # training predictions, test predictions, trained params, batch_norm parameters, hyper parameters
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

    return inputs, targets, test_inputs, test_targets
end


function normalize_data(inputs, test_inputs, norm_mode="none")
    if lowercase(norm_mode) == "standard"
        # normalize training data
        x_mu = mean(inputs, 2)
        x_std = std(inputs, 2)
        inputs = (inputs .- x_mu) ./ (x_std .+ 1e-08)
        # normalize test data
        if size(test_inputs) != (0,0)
            test_inputs = (test_inputs .- x_mu) ./ (x_std .+ 1e-08)
        end
        norm_factors = (x_mu, x_std)
    elseif lowercase(norm_mode) == "minmax"
        # normalize training data
        x_max = maximum(inputs, 2)
        x_min = minimum(inputs, 2)
        inputs = (inputs .- x_min) ./ (x_max .- x_min .+ 1e-08)
        # normalize test data
        if size(test_inputs) != (0,0)
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

    return inputs, test_inputs, norm_factors
end


function setup_model!(mb, hp, tp, bn, dotest, train, test)

#setup mini-batch
    if hp.mb_size < 1
        hp.mb_size = train.n  # use 1 (mini-)batch with all of the examples
    elseif hp.mb_size > train.n
        hp.mb_size = train.n
    elseif mod(train.n, hp.mb_size) != 0
        error("Mini-batch size $(hp.mb_size) does not divide evenly into samples $n.")
    end
    hp.n_mb = Int(train.n / hp.mb_size)  # number of mini-batches
    hp.alphaovermb = hp.alpha / hp.mb_size  # calc once, use in hot loop
    hp.do_batch_norm = hp.n_mb == 1 ? false : hp.do_batch_norm  # no batch normalization for 1 batch

    if hp.mb_size < train.n
        # randomize order of all training samples:
            # labels in training data often in a block, which will make
            # mini-batch train badly because a batch will not contain mix of target labels
        select_index = randperm(train.n)
        train.inputs[:] = train.inputs[:, select_index]
        train.targets[:] = train.targets[:, select_index]
    end

    # set parameters for Momentum or Adam optimization
    if hp.opt == "momentum" || hp.opt == "adam"
        if !(hp.opt_params == [])  # use inputs for opt_params
            # set b1 for Momentum and Adam
            if hp.opt_params[1] > 1.0 || hp.opt_params[1] < 0.5
                warn("First opt_params for momentum or adam should be between 0.5 and 0.999. Using default")
                # nothing to do:  hp.b1 = 0.9 and hp.b2 = 0.999 and hp.ltl_eps = 1e-8
            else
                hp.b1 = hp.opt_params[1] # use the passed parameter
            end
            # set b2 for Adam
            if length(hp.opt_params) > 1
                if hp.opt_params[2] > 1.0 || hp.opt_params[2] < 0.9
                    warn("second opt_params for adam should be between 0.9 and 0.999. Using default")
                else
                    hp.b2 = hp.opt_params[2]
                end
            end
        end
    else
        hp.opt = ""
    end

    hp.do_learn_decay = hp.learn_decay == [1.0, 1.0] ? false :  true

    # dropout parameters: droplim is in hp (Hyper_parameters),
    #    drop_ran_w and drop_filt_w are in mb or train (Model_data)
    # set a droplim for each layer (input layer and output layer will be ignored)
    if hp.dropout
        # fill droplim to match number of hidden layers
        if length(hp.droplim) > length(hp.n_hid)
            hp.droplim = hp.droplim[1:hp.n_hid] # truncate
        elseif length(hp.droplim) < length(hp.n_hid)
            for i = 1:length(hp.n_hid)-length(hp.droplim)
                push!(hp.droplim,hp.droplim[end]) # pad
            end
        end
        hp.droplim = [1.0, hp.droplim..., 1.0] # placeholders for input and output layers
    end

    # debug
    # println("opt params: $(hp.b1), $(hp.b2)")

    # DEBUG see if hyper_parameters set correctly
    # for sym in fieldnames(hp)
    #    println(sym," ",getfield(hp,sym), " ", typeof(getfield(hp,sym)))
    # end

    ##########################################################################
    #  pre-allocate data storage
    ##########################################################################

    preallocate_nn_params!(tp, hp, hp.n_hid, train.in_k, train.n, train.out_k)

    preallocate_feedfwd!(train, tp, train.n, hp.do_batch_norm)

    # feedfwd test data--if test input found
    if dotest
        test.n = size(test.inputs,2)
        preallocate_feedfwd!(test, tp, test.n, hp.do_batch_norm)
    else
        test.n = 0
    end

    # pre-allocate feedfwd mini-batch training data
    preallocate_minibatch!(mb, tp, hp)  
    
    # batch normalization parameters
    if hp.do_batch_norm
        preallocate_batchnorm!(bn, mb, tp.layer_units)
    end

    # debug
    # verify correct dimensions of dropout filter
    # for item in mb.drop_filt_w
    #     println(size(item))
    # end
    # error("that's all folks!....")

end


####################################################################
#  functions to pre-allocate data updated during training loop
####################################################################

# use for test and training data
function preallocate_feedfwd!(dat, tp, n, do_batch_norm)
    # we don't pre-allocate epsilon, grad used during backprop
    dat.a = [dat.inputs]
    dat.z = [zeros(size(dat.inputs))] # not used for input layer  TODO--this permeates the code but not needed
    for i = 2:tp.output_layer-1  # hidden layers
        push!(dat.z, zeros(tp.layer_units[i], n))
        push!(dat.a, zeros(size(dat.z[i])))  #  and up...  ...output layer set after loop
    end
    push!(dat.z, zeros(size(tp.theta[tp.output_layer],1),n))
    push!(dat.a, zeros(size(tp.theta[tp.output_layer],1),n))

    if do_batch_norm  # required for full pass performance stats
        dat.z_norm = deepcopy(dat.z)
        # dat.z_scale = deepcopy(dat.z)  # same size as z, often called "y"
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
    if hp.opt == "momentum" || hp.opt == "adam"
        tp.delta_v_w = [zeros(size(a)) for a in tp.delta_w]
        tp.delta_v_b = [zeros(size(a)) for a in tp.delta_b]
    end
    if hp.opt == "adam"
        tp.delta_s_w = [zeros(size(a)) for a in tp.delta_w]
        tp.delta_s_b = [zeros(size(a)) for a in tp.delta_b]
    end


end


"""
    Pre-allocate these arrays for the training batch--either minibatches or one big batch
    Arrays: epsilon, grad, delta_z_norm, delta_z, drop_ran_w, drop_filt_w
"""
function preallocate_minibatch!(mb, tp, hp)

    mb.epsilon = [zeros(tp.layer_units[l], hp.mb_size) for l in 1:tp.output_layer]
    mb.grad = deepcopy(mb.epsilon)   

    if hp.do_batch_norm
        mb.delta_z_norm = deepcopy(mb.epsilon)  # similar z
        mb.delta_z = deepcopy(mb.epsilon)       # similar z
    end

    #    #debug
    # println("size of pre-allocated mb.delta_z_norm $(size(mb.delta_z_norm))")
    # for i in 1:size(mb.delta_z_norm,1)
    #     println("$i size: $(size(mb.delta_z_norm[i]))")
    # end
    # error("that's all folks....")

    if hp.dropout
        mb.drop_ran_w = deepcopy(mb.epsilon)
        push!(mb.drop_filt_w,fill(true,(2,2))) # for input layer, not used
        for item in mb.drop_ran_w[2:end]
            push!(mb.drop_filt_w,fill(true,size(item)))
        end
    end
end


"""
    Create or update views for the training data in minibatches or one big batch
        Arrays: a, z, z_norm, z_scale, targets  are all fields of struct mb
"""
function update_training_views!(mb::Training_view, train::Model_data, tp::NN_parameters, 
    hp::Hyper_parameters, colrng::UnitRange{Int64})
    n_layers = tp.output_layer

    mb.a = [view(train.a[i],:,colrng) for i = 1:n_layers]
    mb.z = [view(train.z[i],:,colrng) for i = 1:n_layers]
    mb.targets = view(train.targets,:,colrng)  # only at the output layer

    if hp.do_batch_norm
        mb.z_norm = [view(train.z_norm[i],:, colrng) for i = 1:n_layers]
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
    bn.mu = [zeros(i) for i in layer_units]  # same size as bias = no. of layer units
    bn.mu_run = [zeros(i) for i in layer_units]
    bn.stddev = [zeros(i) for i in layer_units]
    bn.std_run = [zeros(i) for i in layer_units]

end

###########################################################################

"""
define and choose functions to be used in neural net training
"""
function setup_functions(units, out_k, opt, classify)

    # all the other functions are module level, e.g. global--make these function variables module level, too
    global unit_function!
    global gradient_function!
    global classify_function!
    global batch_norm_fwd!
    global batch_norm_back!
    global cost_function
    global optimization_function!

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

    if opt == "momentum"
        optimization_function! = momentum!
    elseif opt == "adam"
        optimization_function! = adam!
    else
        optimization_function! = no_optimization
    end

    # set cost function
    cost_function = classify=="regression" ? mse_cost : cross_entropy_cost

end

"""
function feedfwd!(tp, bn, dat, do_batch_norm; istrain)
    modifies a, a_wb, z in place to reduce memory allocations
    send it all of the data or a mini-batch

    feed forward from inputs to output layer predictions
"""
function feedfwd!(tp, bn, dat, hp; istrain=true)

    @fastmath for hl = 2:tp.output_layer-1  # hidden layers

        dat.z[hl][:] = tp.theta[hl] * dat.a[hl-1]

        if hp.do_batch_norm 
            batch_norm_fwd!(bn, dat, hl, istrain)
            unit_function!(dat.z[hl],dat.a[hl])
        else
            dat.z[hl][:] =  dat.z[hl] .+ tp.bias[hl]
            unit_function!(dat.z[hl],dat.a[hl])
        end

        if istrain && hp.dropout  
            dropout!(dat,hp,hl)
        end


    end

    @fastmath dat.z[tp.output_layer][:] = (tp.theta[tp.output_layer] * dat.a[tp.output_layer-1]
        .+ tp.bias[tp.output_layer])  # TODO use bias in the output layer with no batch norm? @inbounds 

    classify_function!(dat.z[tp.output_layer], dat.a[tp.output_layer])  # a = activations = predictions

end


"""
function backprop!(tp, dat, do_batch_norm)
    Argument tp.delta_w holds the computed gradients for weights, delta_b for bias
    Modifies dat.epsilon, tp.delta_w, tp.delta_b in place--caller uses tp.delta_w, tp.delta_b
    Use for training iterations
    Send it all of the data or a mini-batch
    Intermediate storage of dat.a, dat.z, dat.epsilon, tp.delta_w, tp.delta_b reduces memory allocations
"""
function backprop!(tp, bn, dat, hp, t)

    dat.epsilon[tp.output_layer][:] = dat.a[tp.output_layer] .- dat.targets  # @inbounds 
    @fastmath tp.delta_w[tp.output_layer][:] = dat.epsilon[tp.output_layer] * dat.a[tp.output_layer-1]' # 2nd term is effectively the grad for mse  @inbounds 
    @fastmath tp.delta_b[tp.output_layer][:] = sum(dat.epsilon[tp.output_layer],2)  # @inbounds 

    @fastmath for hl = (tp.output_layer - 1):-1:2  # loop over hidden layers
        if hp.do_batch_norm
            gradient_function!(dat.z[hl], dat.grad[hl])
            hp.dropout && (dat.grad[hl] .* dat.drop_filt_w[hl])
            dat.epsilon[hl][:] = tp.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl]  # @inbounds 
            batch_norm_back!(tp, dat, bn, hl)
            tp.delta_w[hl][:] = dat.delta_z[hl] * dat.a[hl-1]'  # @inbounds 
        else
            gradient_function!(dat.z[hl], dat.grad[hl])
            hp.dropout && (dat.grad[hl] .* dat.drop_filt_w[hl])
            dat.epsilon[hl][:] = tp.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl]  # @inbounds 
            tp.delta_w[hl][:] = dat.epsilon[hl] * dat.a[hl-1]'  # @inbounds 
            tp.delta_b[hl][:] = sum(dat.epsilon[hl],2)  #  times a column of 1's = sum(row)
        end

    end

end


###########################################################################

function batch_norm_fwd!(bn, dat, hl, istrain=true)
    in_k,mb = size(dat.z[hl])
    if istrain
        bn.mu[hl][:] = mean(dat.z[hl], 2)          # use in backprop
        bn.stddev[hl][:] = std(dat.z[hl], 2)
        dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu[hl]) ./ bn.stddev[hl]  # normalized: 'aka' xhat or zhat  @inbounds 
        dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: 'aka' y  @inbounds 
        bn.mu_run[hl][:] = (  bn.mu_run[hl][1] == 0.0 ? bn.mu[hl] :  # @inbounds 
            0.9 .* bn.mu_run[hl] .+ 0.1 .* bn.mu[hl]  )
        bn.std_run[hl][:] = (  bn.std_run[hl][1] == 0.0 ? bn.stddev[hl] :  # @inbounds 
            0.9 .* bn.std_run[hl] + 0.1 .* bn.stddev[hl]  )
    else  # predictions with existing parameters
        dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu_run[hl]) ./ bn.std_run[hl]  # normalized: 'aka' xhat or zhat  @inbounds 
        dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: 'aka' y  @inbounds 
    end
end


function batch_norm_back!(tp, dat, bn, hl)
    d,mb = size(dat.epsilon[hl])
    bn.delta_bet[hl][:] = sum(dat.epsilon[hl], 2)
    bn.delta_gam[hl][:] = sum(dat.epsilon[hl] .* dat.z_norm[hl], 2)

    # debug
    # println("size of pre-allocated dat.delta_z_norm $(size(dat.delta_z_norm))")
    # for i in 1:size(dat.delta_z_norm,1)
    #     println("$i size: $(size(dat.delta_z_norm[i]))")
    # end
    # error("that's all folks....")

    # debug
    # println("size of pre-allocated mb.epsilon $(size(dat.epsilon))")
    # for i in 1:size(dat.epsilon,1)
    #     println("$i size: $(size(dat.epsilon[i]))")
    # end
    # error("that's all folks....")

    dat.delta_z_norm[hl][:] = bn.gam[hl] .* dat.epsilon[hl]  # @inbounds 

    dat.delta_z[hl][:] = (                               # @inbounds 
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


function gather_stats!(i, plotdef, train, test, tp, bn, cost_function, train_n, test_n, hp)

    if plotdef["plot_switch"]["Training"]
        feedfwd!(tp, bn, train, hp, istrain=false)

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
        feedfwd!(tp, bn, test, hp, istrain=false)
        if plotdef["plot_switch"]["Cost"]
            cost = cost_function(test.targets,
                test.a[tp.output_layer], test.n, tp.theta, hp, tp.output_layer)
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
    predictions = feedfwd!(tp, bn, dat, do_batch_norm, istrain=false)
end

# predict:  
#     do one pass for a set of data
#     need to do training or test data in own pass--data is data at this stage
# read the model data
# read the hyperparameters, trained parameters, batch_norm hyper_parameters
# set up feedforward:  
#     pre-allocate data
#     pre-allocate feedfwd
#     define needed functions
# feedforward


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

end  # module GeneralNN
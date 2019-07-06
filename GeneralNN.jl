#DONE


#TODO
#   simplify use of input parameters for minibatch training
#   replace datalist with a dict and get rid of duplicate splitting logic?
#   use goodness function to hold either accuracy or r_squared
#   fix preallocation for test set--no backprop arrays needed?
#   is it worth it to pre-allocate expf when using softmax classification?
#   sort out what preallocation is needed for Batch and batch with batchnorm
#   Switch to an explicit layer based approach to allow layers to be different
#   fix dropout: should delta_theta be averaged using number of included units?
        #   are we scaling units during training?
#   set an explicit data size variable for both test and train for preallocation?
#   new minibatch selection is slow because it is non-contiguous: elim dynamic
            # shuffle slicing--require pre-shuffle to enable linear indexing of
            # views.
            # Also, preallocation of views no longer typed.
#   setup separate data structures for sparse and views on sparse?          
#   accuracy for logistic or other single class classifiers should allow -1 or 0 for "bad"
#   TODO use bias in the output layer with no batch norm?  YES
#   implement precision and recall
#   stats on individual regression parameters
#   change to layer based api, with simplified api as alternate front-end
#   separate reg from the cost calculation and the parameter updates
#   do check on existence of matfname file and that type is .mat
#   implement one vs. all for logistic classification with multiple classes
#   factor out extract data--provide as utility; start process with datafiles
#   is dropout dropping the same units on backprop as feedfwd?  seems like it but results are still poor
#   set a directory for training stats (keep out of code project directory)
#   there is no reason for views on backprop data--always the size of minibatch--is it ok to reuse same
    #   arrays--e.g., do we need a cache or to have values across the entire dataset?
#   revise initialization and make it depend on type of layer unit???
#   try different versions of ensemble predictions_vector
#   augment MINST data by perturbing the images
#   separate plotdef from plot descriptive data
#   check for type stability: @code_warntype pisum(500,10000)
#   still lots of memory allocations despite the pre-allocation
        # You can devectorize r -= d[j]*A[:,j] with r .= -.(r,d[j]*A[:.j]) 
        #        to get rid of some more temporaries. 
        # As @LutfullahTomak said sum(A[:,j].*r) should devectorize as dot(view(A,:,j),r) 
        #        to get rid of all of the temporaries in there. 
        # To use an infix operator, you can use \cdot, as in view(A,:,j)⋅r.
#   figure out memory use between train set and minibatch set
#   implement a gradient checking function with option to run it
#   Convolutional networks
    #   convolutional layers
    #   pooling layers
# Recurrent Neural networks
#   implement early stopping


"""
Module GeneralNN:

Includes the following functions to run directly:

- train_nn() -- train neural networks for up to 9 hidden layers
- test_score() -- cost and accuracy given test data and saved theta
- save_params() -- save all model parameters
- load_params() -- load all model parameters
- predictions_vector() -- predictions given x and saved theta
- accuracy() -- calculates accuracy of predictions compared to actual labels
- extract_data() -- extracts data for MNIST from matlab files
- normalize_inputs() -- normalize via standardization (mean, sd) or minmax
- normalize_replay!() --
- nnpredict() --  calculate predictions from previously trained parameters and new data
- display_mnist_digit() --  show a digit
- wrong_preds() --  return the indices of wrong predictions against supplied correct labels
- right_preds() --  return the indices of correct predictions against supplied correct labels
- plot_output() --  plot learning, cost for training and test for previously saved training runs

To use, include() the file.  Then enter using .GeneralNN to use the module.

These data structures are used to hold parameters and data:

- NN_weights holds theta, bias, delta_w, delta_b, theta_dims, output_layer, k
- Model_data holds inputs, targets, a, z, z_norm, epsilon, gradient_function
- Batch_norm_params holds gam (gamma), bet (beta) for batch normalization and intermediate
    data used for backprop with batch normalization: delta_gam, delta_bet, 
    delta_z_norm, delta_z, mu, stddev, mu_run, std_run
- Hyper_parameters holds user input hyper_parameters and some derived parameters.
- Batch_view holds views on model data to hold the subset of data used in minibatches. 
  (This is not exported as there is no way to use it outside of the backprop loop.)

"""
module GeneralNN


# ----------------------------------------------------------------------------------------

# data structures for neural network
export 
    NN_weights, 
    Model_data, 
    Batch_norm_params, 
    Hyper_parameters

# functions you can use
export 
    train_nn, 
    test_score, 
    save_params, 
    load_params, 
    accuracy,
    extract_data, 
    shuffle_data!,
    normalize_inputs!, 
    nnpredict,
    display_mnist_digit,
    wrong_preds,
    right_preds,
    plot_output

using MAT
using JLD2
using FileIO
using Statistics
using Random
using Printf
using Dates
using LinearAlgebra
using JSON
using SparseArrays

using Plots
# plotlyjs()  # PlotlyJS backend to local electron window
pyplot()

import Measures: mm # for some plot dimensioning

include("training_loop.jl")
include("layer_functions.jl")
include("nn_data_structs.jl")
include("setup_model.jl")
include("utilities.jl")

const l_relu_neg = .01  # makes the type constant; value can be changed.

# ----------------------------------------------------------------------------------------


"""
function train_nn(train_x, train_y, test_x, test_y, epochs::Int64, n_hid::Array{Int64,1}; alpha::Float64=0.35,
    mb_size::Int64=0, lambda::Float64=0.01, classify::String="softmax", norm_mode::String="none",
    opt::String="", opt_params::Array{Float64,1}=[0.9,0.999], units::String="sigmoid", do_batch_norm::Bool=false,
    reg::String="L2", dropout::Bool=false, droplim::Array{Float64,1}=[0.5], plots::Array{String,1}=["Training", "Learning"],
    learn_decay::Array{Float64,1}=[1.0, 1.0])

Train sigmoid/softmax neural networks up to 11 layers.  Detects
number of output labels from data. Detects number of features from data for input units.
Enables any size mini-batch (last batch will be smaller if minibatch size doesn't divide evenly
into number of examples).  Plots learning and cost outcomes by epoch for training and test data.

This is a front-end function that verifies all inputs and calls run_training(). A convenience method allows 
all of the input parameters to be read from a JSON file.  This is further explained below.


    returns a dict containing these keys:  
        train_inputs      ::= after any normalization
        train_targets     ::= matches original inputs
        train_preds       ::= using final values of trained parameters
        test_inputs       ::= after any normalization
        test_targets      ::= matches original inputs
        test_preds        ::= using final values of trained parameters
        nn_params         ::= struct that holds all trained parameters
        batch_norm_params ::= struct that holds all batch_norm parameters
        hyper_params      ::= all hyper parameters used to control training


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
        opt             ::= one of "Momentum", "RMSProp", "Adam" or "".  default is blank string "".
        opt_params      ::= parameters used by Momentum, RMSProp, or Adam
                           Momentum, RMSProp: one floating point value as [.9] (showing default)
                           Adam: 2 floating point values as [.9, .999] (showing defaults)
                           Note that epsilon is ALWAYS set to 1e-8
                           To accept defaults, don't input this parameter or use []
        classify        ::= "softmax", "sigmoid", or "regression" for only the output layer
        units           ::= "sigmoid", "l_relu", "relu", "tanh" for non-linear activation of all hidden layers
        plots           ::= determines training results collected and plotted
                            any choice of ["Learning", "Cost", "Training", "Test"];
                            for no plots use [""] or ["none"]
        reg             ::= type of regularization, must be one of "L1", "L2", "Maxnorm", ""
        maxnorm_lim     ::= array of limits set for hidden layers + output layer
        dropout         ::= true to use dropout network or false
        droplim         ::= array of values between 0.5 and 1.0 determines how much dropout for
                            hidden layers and input layer (ex: [0.8] or [0.8,0.9, 1.0]).  A single
                            value will be applied to all hidden layers.  If fewer values than layers, then the
                            last value extends to remaining layers.
        learn_decay     ::= array of 2 float values:  first is > 0.0 and <= 1.0 which is pct. reduction of 
                            learning rate; second is >= 1.0 and <= 10.0 for number of times to 
                            reduce learning decay_rate
                            [1.0, 1.0] signals don't do learning decay
        plot_now        ::Bool.  If true, plot training stats and save the plotdef that contains stats 
                            gathered while running the training.  You can plot the file separately, later.
        sparse          ::Bool. If true, input data will be treated and maintained as SparseArrays.
        initializer     ::= "xavier" or "zero" used to set how weights, not including bias, are initialized.
        quiet           ::Bool. Suppress progress messages during preparation and training.
        shuffle         ::Bool. Determines if values are randomized when assigned to minibatches. Very slow!

This method allows all input parameters to be supplied by a JSON file:

    function train_nn(argsjsonfile::String, errorcheck::Bool=false)

    The json file need only contain argument values that you desire to set
    differently than the defaults. Except for the last arguments, all will
    be set as hyper_parameters.

    Here is an example of a correct json file containing every permitted argument:

        {
            "units":  "relu",
            "alpha":   0.74,
            "lambda":  0.000191,
            "n_hid": [100],
            "reg":  "L2",
            "maxnorm_lim": [],
            "classify": "softmax",
            "dropout": false,
            "droplim": [1.0,0.8,1.0],
            "epochs":  24,
            "mb_size_in":   50,
            "norm_mode":   "none",
            "opt":   "adam",
            "opt_params": [0.9, 0.999],   
            "learn_decay": [0.5,4.0],
            "dobatch": true,
            "do_batch_norm":  true,
            "sparse": false,
            "initializer": "xavier",   
            "quiet": true,
            "shuffle": false
            "plots": ["Train", "Learning", "Test"],
            "plot_now": true                    # not a hyper_parameter
        }

    If errorcheck is set to true the JSON file is checked:
       1) To make sure all required arguments are present; this is true even
          though the function that will be called provides valid defaults.
       2) To make sure that all argument names are valid.
    If any errors are found, neural network training is not run.
"""
function train_nn(datalist, epochs::Int64, n_hid::Array{Int64,1}; 
    alpha::Float64=0.35, mb_size_in::Int64=0, lambda::Float64=0.01, classify::String="softmax", 
    norm_mode::String="none", opt::String="", opt_params::Array{Float64,1}=[0.9,0.999], 
    units::String="sigmoid", dobatch=false, do_batch_norm::Bool=false, reg::String="L2", 
    maxnorm_lim::Array{Float64,1}=Float64[], dropout::Bool=false, droplim::Array{Float64,1}=[0.5], 
    plots::Array{String,1}=["Training", "Learning"], learn_decay::Array{Float64,1}=[1.0, 1.0], 
    plot_now::Bool=true, sparse::Bool=false, initializer::String="xavier", quiet=true, shuffle=false)

    # validate hyper_parameters and put into struct hp
    hp = validate_hyper_parameters(units, alpha, lambda, n_hid, reg, maxnorm_lim, classify, dropout,
            droplim, epochs, mb_size_in, norm_mode, opt, opt_params, learn_decay, dobatch,
            do_batch_norm, sparse, initializer, quiet, shuffle, plots)
            
    validate_datafiles(datalist) # no return; errors out if errors
 
    run_training(datalist, hp; plot_now=plot_now);
end


function train_nn(datalist, argsjsonfile::String, errorcheck::Bool=false)
################################################################################
#   This method gets input arguments from a JSON file. This method does no
#   error checking except, optionally, for valid argnames or missing required args. 
#   This method corrects types that are not passed properly by JSON.
################################################################################

    argstxt = read(argsjsonfile, String)
    argsdict = JSON.parse(argstxt)
    inputargslist = keys(argsdict)

    errorcheck && check_json_inputs(inputargslist)

    # collect individual required args
    epochs = pop!(argsdict, "epochs")
    n_hid = Int64.(pop!(argsdict, "n_hid"))

    # convert  JSON array type Any to appropriate Julia type
    if "plots" in inputargslist
        argsdict["plots"] = String.(argsdict["plots"])
    end
    if "learn_decay" in inputargslist
        argsdict["learn_decay"] = Float64.(argsdict["learn_decay"])
    end
    if "opt_params" in inputargslist
        argsdict["opt_params"] = Float64.(argsdict["opt_params"])
    end
    if "droplim" in inputargslist
        argsdict["droplim"] = Float64.(argsdict["droplim"])
    end
    if "maxnorm_lim" in inputargslist
        argsdict["maxnorm_lim"] = Float64.(argsdict["maxnorm_lim"])
    end

    train_nn( 
             datalist, epochs, n_hid; 
             Dict(   zip(Symbol.(keys(argsdict)),values(argsdict))    )... # Don't forget the ... splat!
             )
end



"""
function run_training(datalist, hp; plot_now=true)

This is the one function and only method that really does the work.  It runs
the model training or as many frameworks refer to it, "fits" the model.
"""
function run_training(datalist, hp; plot_now=true)

    !hp.quiet && println("Setting up model beginning")
    # seed random number generator.  For runs of identical models the same weight initialization
    # will be used, given the number of parameters to be estimated.  Enables better comparisons.
    Random.seed!(70653)  # seed int value is meaningless

    if size(datalist,1) == 4
        train_x = datalist[1]
        train_y = datalist[2]
        test_x = datalist[3]
        test_y = datalist[4]
        dotest = size(test_x, 1) > 0  # there is testing data -> true, else false
        if !dotest
            error("Test data inputs are empty. Rerun without passing test data inputs at all.")
        end
    elseif size(datalist, 1) == 2
        train_x = datalist[1]
        train_y = datalist[2]
        dotest = false
    else
        error("Datalist input contained wrong number of inputs")
    end

    ##################################################################################
    #   setup model: data structs, many control parameters, functions,  memory pre-allocation
    #################################################################################



    # instantiate data containers
    !hp.quiet && println("Instantiate data containers")

    train = Model_data()  # train holds all the training data and layer inputs/outputs
    dotest && (test = Model_data())   # for test--but there is no training, just prediction
    hp.dobatch && (mb = Batch_view())  # layer data for mini-batches: = Batch_slice()    or  = Batch_view
    nnp = NN_weights()  # neural network trained parameters
    bn = Batch_norm_params()  # do we always need the data structure to run?  yes--TODO fix this

    !hp.quiet && println("Set input data aliases to model data structures")
    if dotest
        train.inputs, train.targets, test.inputs, test.targets = train_x, train_y, test_x, test_y
    else
        train.inputs, train.targets = train_x, train_y
    end
    !hp.quiet && println("Alias to model data structures completed")

    # set some useful variables
    train.in_k, train.n = size(train_x)  # number of features in_k (rows) by no. of examples n (columns)
    dotest && ((test.in_k, test.n) = size(test_x))
    train.out_k = size(train_y,1)  # number of output units

    #  optimization parameters, minibatch, 
    hp.dobatch && setup_model!(mb, hp, nnp, bn, train)
    hp.dobatch || setup_model!(hp, nnp, bn, train)

    # normalize data
    if !(hp.norm_mode == "" || lowercase(hp.norm_mode) == "none")
        nnp.norm_factors = normalize_inputs!(train.inputs, hp.norm_mode)
        dotest && normalize_inputs!(test.inputs, nnp.norm_factors, hp.norm_mode) 
    end

    # preallocate data storage
        !hp.quiet && println("Pre-allocate storage starting")
        preallocate_nn_params!(nnp, hp, train.in_k, train.n, train.out_k)
        preallocate_data!(train, nnp, train.n, hp)
        hp.dobatch && preallocate_minibatch!(mb, nnp, hp)
        hp.do_batch_norm && preallocate_batchnorm!(bn, mb, nnp.k)
        dotest && preallocate_data!(test, nnp, test.n, hp, istrain=false)
        !hp.quiet && println("Pre-allocate storage completed")

    # choose layer functions and cost function based on inputs
    setup_functions!(hp, train)

    # statistics for plots and history data
    plotdef = setup_plots(hp, dotest)

    !hp.quiet && println("Training setup complete")
    
    
    ##########################################################
    #   neural network training loop
    ##########################################################
    datalist = dotest ? [train, test] : [train]
    
    if hp.dobatch
        training_time = training_loop(hp, datalist, mb, nnp, bn, plotdef)
    else
        training_time = training_loop(hp, datalist, nnp, plotdef)
    end
    
    # save, print and plot training statistics after all epochs
    output_stats(datalist, nnp, bn, hp, training_time, plotdef, plot_now)

    #  return train inputs, train targets, train predictions, test predictions, trained parameters, batch_norm parms., hyper parms.
    ret = Dict(
                "train_inputs" => train_x, 
                "train_targets"=> train_y, 
                "train_preds" => train.a[nnp.output_layer], 
                "nn_params" => nnp, 
                "batchnorm_params" => bn, 
                "hyper_params" => hp
                )
    dotest &&   begin
                    ret["test_inputs"] = test.inputs 
                    ret["test_targets"] = test.targets 
                    ret["test_preds"] = test.a[nnp.output_layer]  
                end 
    return ret

end # run_training_core, method with test data


function validate_datafiles(datalist)
   if size(datalist,1) == 4
        train_x = datalist[1]
        train_y = datalist[2]
        test_x = datalist[3]
        test_y = datalist[4]
        dotest = size(test_x, 1) > 0  # there is testing data -> true, else false
    elseif size(datalist, 1) == 2
        train_x = datalist[1]
        train_y = datalist[2]
        dotest = false
    else
        error("Datalist input contained wrong number of input arrays")
    end
    
    # training data
    (train_m, train_n) = size(train_x)
    (try_m, try_n) = size(train_y)
    if train_m >= train_n
        error("No. of features is greater than no. of samples. Probably the training array must be tranposed.")
    end
    if try_m >= try_n
        error("No. of output labels is greater than no. of samples. Probably the label array must be transposed.")
    end
    if try_n != train_n
        error("No. of training inputs does not match no. of training label outputs.")
    end
    
    # test or validation data
    if dotest
        (test_m, test_n) = size(test_x)
        (testy_m, testy_n) = size(test_y)
        if test_m >= test_n
            error("No. of features is greater than no. of samples. Probably the test array must be tranposed.")
        end
        if testy_m >= testy_n
            error("No. of test output labels is greater than no. of samples. Probably the test label array must be transposed.")
        end
        if testy_n != test_n
            error("No. of test inputs does not match no. of test label outputs.")
        end
        
        if train_m != test_m
            error("No. of training features does not match test features.")
        end   
    end
end


function validate_hyper_parameters(units, alpha, lambda, n_hid, reg, maxnorm_lim, classify, dropout,
    droplim, epochs, mb_size_in, norm_mode, opt, opt_params, learn_decay, dobatch,
    do_batch_norm, sparse, initializer, quiet, shuffle, plots)

    !quiet && println("Validate input parameters")

    ################################################################################
    #   This is a front-end function that verifies all inputs and returns the hyper_parameters struct
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
        @warn("Alpha learning rate set too small. Proceeding...")
        # alpha = 0.35
    elseif alpha > 9.0
        @warn("Alpha learning rate set too large. Proceeding...")
        # alpha = 0.35
    end

    if mb_size_in < 0
        error("Input mb_size must be an integer greater or equal to 0")
    end

    classify = lowercase(classify)
    if !in(classify, ["softmax", "sigmoid", "regression", "logistic"])
        error("classify must be \"softmax\", \"sigmoid\", \"logistic\" or \"regression\".")
    end

    norm_mode = lowercase(norm_mode)
        if !in(norm_mode, ["", "none", "standard", "minmax"])
            @warn("Invalid norm mode: $norm_mode, using \"none\".")
            norm_mode = ""
        end

    units = lowercase(units)
        if !in(units, ["l_relu", "sigmoid", "relu", "tanh"])
            @warn("units must be \"relu,\" \"l_relu,\" \"tanh\" or \"sigmoid\". Setting to default \"sigmoid\".")
            units = "sigmoid"
        end

    if in(units, ["l_relu", "relu"])
        if (norm_mode=="" || norm_mode=="none") && !do_batch_norm
            @warn("Better results obtained with relu using input and/or batch normalization. Proceeding...")
        end
    end

opt = lowercase(opt)  # match title case for string argument
    if !in(opt, ["momentum", "adam", "rmsprop", ""])
        @warn("opt must be \"momentum\", \"rmpsprop\" or \"adam\" or \"\" (nothing).  Setting to \"\" (nothing).")
        opt = ""
    end

    if in(opt, ["momentum", "adam", "rmsprop"])
        if size(opt_params) == (2,)
            if opt_params[1] > 1.0 || opt_params[1] < 0.5
                @warn("First opt_params for momentum, rmsprop or adam should be between 0.5 and 0.999. Using default")
                opt_params = [0.9, opt_params[2]]
            end
            if opt_params[2] > 1.0 || opt_params[2] < 0.8
                @warn("second opt_params for adam should be between 0.8 and 0.999. Using default")
                opt_params = [opt_params[1], 0.999]
            end
        else
            @warn("opt_params must be 2 element array with values between 0.9 and 0.999. Using default")
            opt_params = [0.9, 0.999]
        end
    end

    reg = titlecase(lowercase(reg))
        if !in(reg, ["L1", "L2", "Maxnorm", ""])
            @warn("reg must be \"L1\", \"L2\", \"Maxnorm\" or \"\" (nothing). Setting to default \"L2\".")
            reg = "L2"
        end

    if dropout
        if !all([(c>=.2 && c<=1.0) for c in droplim])
            error("droplim values must be between 0.2 and 1.0. Quitting.")
        end
    end

    # lambda
    if reg == "L2"  || reg == "L1"
        if lambda < 0.0  # set reg = "" relu with batch_norm
            @warn("Lambda regularization rate must be positive floating point value. Setting to 0.01")
            lamba = 0.01
        elseif lambda > 5.0
            @warn("Lambda regularization rate probably set too large. Proceeding with your input.")
            # lambda == 5.0
        end
    end

    learn_decay = 
        if size(learn_decay) != (2,)
            @warn("learn_decay must be a vector of 2 numbers. Using no learn_decay")
            [1.0, 1.0]
        elseif !(learn_decay[1] >= 0.0 && learn_decay[1] <= 1.0)
            @warn("First value of learn_decay must be >= 0.0 and < 1.0. Using no learn_decay")
            [1.0, 1.0]
        elseif !(learn_decay[2] >= 1.0 && learn_decay[2] < 10.0)
            @warn("Second value of learn_decay must be >= 1.0 and <= 10.0. Using no learn_decay")
            [1.0, 1.0]
        else
            [learn_decay[1], floor(learn_decay[2])]
        end

    valid_plots = ["train", "test", "learning", "cost", "none", "", "epoch", "batch"]
    plots = lowercase.(plots)  # user input doesn't have use perfect case
    new_plots = [pl for pl in valid_plots if in(pl, plots)] # both valid and input by the user
    plots = 
        if sort(new_plots) != sort(plots) # the plots list included something not in valid_plots
            @warn("Plots argument can only include \"train\", \"test\", \"learning\", \"epoch\", \"batch\" and \"cost\" or \"none\" or \"\".
                \nProceeding no plots [\"none\"].")
            ["none"]
        else
            new_plots
        end

    initializer = lowercase(initializer)
        if !in(initializer, ["zero", "xavier"])
            @warn("initializer must be \"zero\" or \"xavier\". Setting to default \"xavier\".")
            initializer = "xavier"
        end

    hp = Hyper_parameters()  # hyper_parameters constructor:  sets defaults
    # update Hyper_parameters with user inputs; others set in function setup_model!
        hp.units = units
        hp.alpha = alpha
        hp.lambda = lambda
        hp.n_hid = n_hid
        hp.reg = reg
        hp.maxnorm_lim = maxnorm_lim
        hp.classify = classify
        hp.dropout = dropout
        hp.droplim = droplim
        hp.epochs = epochs
        hp.mb_size_in = mb_size_in
        hp.norm_mode = norm_mode
        hp.opt = opt
        hp.opt_params = opt_params
        hp.learn_decay = learn_decay
        hp.dobatch = dobatch
        hp.do_batch_norm = do_batch_norm
        hp.sparse = sparse
        hp.initializer = initializer
        hp.quiet = quiet
        hp.shuffle = shuffle
        hp.plots = plots


    !quiet && println("Validation of input parameters completed")

    return hp
end


function check_json_inputs(inputargslist)

    validargnames = [
                     "matfname", "epochs", "n_hid", "alpha", "mb_size_in", "lambda",
                     "classify", "norm_mode", "opt", "opt_params", "units", "dobatch", "do_batch_norm",
                     "reg", "maxnorm_lim", "dropout", "droplim", "plots", "learn_decay", "plot_now", 
                     "sparse", "initializer", "quiet", "shuffle"
                     ]
    requiredargs = ["matfname", "epochs", "n_hid"]

    # check for missing required args
    errnames = []            
    for argname in requiredargs
        if !(argname in inputargslist)
            push!(errnames, argname)
        end
    end
    if !isempty(errnames)
        println("Input file missing required arguments:")
        println(errnames)
        error("Stopped with invalid inputs")
    end

    # check for invalid arg names
    errnames = []
    for argname in inputargslist
        if !(argname in validargnames)
            push!(errnames, argname)
        end
    end
    if !isempty(errnames)
        println("Input file contained invalid argument names:")
        println(errnames)
        println("Valid argument names are:")
        println(validargnames)
        error("Stopped with invalid inputs")
    end

    println("All input args valid, proceeding...")

end


end  # module GeneralNN
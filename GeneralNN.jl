#DONE



#TODO
#   eliminate size() in loops
#   there is no reason for views on backprop data--always the size of minibatch
#   everything working now:  try @inbounds to see what effect it has
#   run checks for type stability
#   revise initialization and make it depend on type of layer unit
#   try different versions of ensemble predictions_vector
#   allow dropout to drop from the input layer
#   augment data by perturbing the images
#   don't create plotdef if not plotting
#   try batch norm with minmax normalization
#   cleaning up batch norm is complicated:
#       affects feedfwd, backprop, pre-allocation (several), momentum, adam search for if [!hp.]*do_batch_norm
#       type dispatch on bn:  either a struct or a bool to eliminate if tests all over to see if we batch normalize
#   check for type stability: @code_warntype pisum(500,10000)
#   still lots of memory allocations despite the pre-allocation
        # You can devectorize r -= d[j]*A[:,j] with r .= -.(r,d[j]*A[:.j]) 
        #        to get rid of some more temporaries. 
        # As @LutfullahTomak said sum(A[:,j].*r) should devectorize as dot(view(A,:,j),r) 
        #        to get rid of all of the temporaries in there. 
        # To use an infix operator, you can use \cdot, as in view(A,:,j)⋅r.
#   stats on individual regression parameters
#   figure out memory use between train set and minibatch set
#   fix predictions
#   make affine units a separate layer with functions for feedfwd, gradient and test--do we need to?
#   try "flatscale" x = x / max(x)
#   implement a gradient checking function with option to run it
#   convolutional layers
#   pooling layers
#   implement early stopping
#   separate plots data structure from stats data structure?

#  __precompile__()  # not obvious this does anything at all as no persistent cache is created

"""
Module GeneralNN:

Includes the following functions to run directly:

- train_nn() -- train sigmoid/softmax neural networks for up to 9 hidden layers
- test_score() -- cost and accuracy given test data and saved theta
- save_params() -- save all model parameters
- load_params() -- load all model parameters
- predictions_vector() -- predictions given x and saved theta
- accuracy() -- calculates accuracy of predictions compared to actual labels
- extract_data() -- extracts data for MNIST from matlab files
- normalize_inputs() -- normalize via standardization (mean, sd) or minmax

To use, include() the file.  Then enter using GeneralNN to use the module.

These data structures are used to hold parameters and data:

- NN_parameters holds theta, bias, delta_w, delta_b, theta_dims, output_layer, layer_units
- Model_data holds inputs, targets, a, z, z_norm, epsilon, gradient_function
- Batch_norm_params holds gam (gamma), bet (beta) for batch normalization and Intermediate
data used for training: delta_gam, delta_bet, delta_z_norm, delta_z, mu, stddev, mu_run, std_run

"""
module GeneralNN


# ----------------------------------------------------------------------------------------

# data structures for neural network
export 
    NN_parameters, 
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
    normalize_inputs, 
    normalize_replay!, 
    nnpredict,
    display_mnist_digit,
    wrong_preds,
    right_preds,
    plot_output

using MAT
using JLD2
using FileIO

# new plotting approach
using Plots
plotlyjs()  # PlotlyJS backend to local electron window

import Measures: Length, AbsoluteLength, Measure, BoundingBox, mm, cm, inch, pt, width, height, w, h


include("layer_functions.jl")
include("nn_data_structs.jl")
include("setup_functions.jl")
include("utilities.jl")

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
Enables any size mini-batch that divides evenly into number of examples.  Plots learning 
and cost outcomes by iteration for training and test data.

This is a front-end function that verifies all inputs and calls run_training().

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
        units           ::= "sigmoid", "l_relu", "relu", "tanh" for non-linear activation of all hidden layers
        plots           ::= determines training results collected and plotted
                            any choice of ["Learning", "Cost", "Training", "Test"];
                            for no plots use [""] or ["none"]
        reg             ::= type of regularization, must be one of "L2", ""
        dropout         ::= true to use dropout network or false
        droplim         ::= array of values between 0.5 and 1.0 determines how much dropout for
                            hidden layers and output layer (ex: [0.8] or [0.8,0.9, 1.0]).  A single
                            value will be applied to all layers.  If fewer values than layers, then the
                            last value extends to remaining layers.
        learn_decay     ::= array of 2 float values:  first is > 0.0 and <= 1.0 which is pct. reduction of 
                            learning rate; second is >= 1.0 and <= 10.0 for number of times to 
                            reduce learning decay_rate
                            [1.0, 1.0] signals don't do learning decay
        plot_now        ::Bool.  If true, plot training stats and save the plotdef that contains stats 
                            gathered while running the training.  You can plot the file separately, later.


"""
function train_nn(matfname::String, epochs::Int64, n_hid::Array{Int64,1}; alpha::Float64=0.35,
    mb_size_in::Int64=0, lambda::Float64=0.01, classify::String="softmax", norm_mode::String="none",
    opt::String="", opt_params::Array{Float64,1}=[0.9,0.999], units::String="sigmoid", do_batch_norm::Bool=false,
    reg::String="L2", dropout::Bool=false, droplim::Array{Float64,1}=[0.5], plots::Array{String,1}=["Training", "Learning"],
    learn_decay::Array{Float64,1}=[1.0, 1.0], plot_now::Bool=true)

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
    elseif alpha > 9.0
        warn("Alpha learning rate set too large. Setting to defaut 0.35")
        alpha = 0.35
    end

    if mb_size_in < 0
        error("Input mb_size must be an integer greater or equal to 0")
    end

    classify = lowercase(classify)
    if !in(classify, ["softmax", "sigmoid", "regression"])
        error("classify must be \"softmax\", \"sigmoid\" or \"regression\".")
    end

    norm_mode = lowercase(norm_mode)
        if !in(norm_mode, ["", "none", "standard", "minmax"])
            warn("Invalid norm mode: $norm_mode, using \"none\".")
            norm_mode = ""
        end

    units = lowercase(units)
        if !in(units, ["l_relu", "sigmoid", "relu", "tanh"])
            warn("units must be \"relu,\" \"l_relu,\" \"tanh\" or \"sigmoid\". Setting to default \"sigmoid\".")
            units = "sigmoid"
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
    end


    if in(opt, ["momentum", "adam"])
        if size(opt_params) == (2,)
            if opt_params[1] > 1.0 || opt_params[1] < 0.5
                warn("First opt_params for momentum or adam should be between 0.5 and 0.999. Using default")
                opt_params = [0.9, opt_params[2]]
            end
            if opt_params[2] > 1.0 || opt_params[2] < 0.8
                warn("second opt_params for adam should be between 0.8 and 0.999. Using default")
                opt_params = [opt_params[1], 0.999]
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

    # lambda
    if reg == "L2"
        if lambda < 0.0  # set reg = "" relu with batch_norm
            warn("Lambda regularization rate must be positive floating point value. Setting to 0.01")
            lamba = 0.01
        elseif lambda > 5.0
            warn("Lambda regularization rate set too large. Setting to max of 5.0")
            lambda == 5.0
        end
    end

    learn_decay = 
        if size(learn_decay) != (2,)
            warn("learn_decay must be a vector of 2 numbers. Using no learn_decay")
            [1.0, 1.0]
        elseif !(learn_decay[1] >= 0.0 && learn_decay[1] <= 1.0)
            warn("First value of learn_decay must be >= 0.0 and < 1.0. Using no learn_decay")
            [1.0, 1.0]
        elseif !(learn_decay[2] >= 1.0 && learn_decay[2] < 10.0)
            warn("Second value of learn_decay must be >= 1.0 and <= 10.0. Using no learn_decay")
            [1.0, 1.0]
        else
            [learn_decay[1], floor(learn_decay[2])]
        end

    valid_plots = ["Training", "Test", "Learning", "Cost", "None", ""]
    plots = titlecase.(lowercase.(plots))  # user input doesn't have use perfect case
    new_plots = [pl for pl in valid_plots if in(pl, plots)] # both valid and input by the user
    plots = 
        if sort(new_plots) != sort(plots) # the plots list included something not in valid_plots
            warn("Plots argument can only include \"Training\", \"Test\", \"Learning\" and \"Cost\" or \"None\" or \"\".
                \nProceeding no plots [\"None\"].")
            ["None"]
        else
            new_plots
        end

    run_training(matfname, epochs, n_hid,
        plots=plots, reg=reg, alpha=alpha, mb_size_in=mb_size_in, lambda=lambda,
        opt=opt, opt_params=opt_params, classify=classify, dropout=dropout, droplim=droplim,
        norm_mode=norm_mode, do_batch_norm=do_batch_norm, units=units, learn_decay=learn_decay,
        plot_now=plot_now);
end


function run_training(matfname::String, epochs::Int64, n_hid::Array{Int64,1};
    plots::Array{String,1}=["Training", "Learning"], reg="L2", alpha=0.35,
    mb_size_in=0, lambda=0.01, opt="", opt_params=[], dropout=false, droplim=[0.5],
    classify="softmax", norm_mode="none", do_batch_norm=false, units="sigmoid",
    learn_decay::Array{Float64,1}=[1.0, 1.0], plot_now=true)


    # seed random number generator.  For runs of identical models the same weight initialization
    # will be used, given the number of parameters to be estimated.  Enables better comparisons.
    srand(70653)  # seed int value is meaningless


    ##################################################################################
    #   setup model: data structs, many control parameters, functions,  pre-allocation
    #################################################################################

    # instantiate data containers
    train = Model_data()  # train holds all the training data and layer inputs/outputs
    test = Model_data()   # for test--but there is no training, just prediction
    mb = Training_view()  # layer data for mini-batches: as views on training data or arrays
    nnp = NN_parameters()  # trained parameters
    bn = Batch_norm_params()  # do we always need the data structure to run?  yes--TODO fix this

    hp = Hyper_parameters()  # hyper_parameters:  sets defaults
    # update Hyper_parameters with user inputs--more below based on other logic
        hp.units = units
        hp.alpha = alpha
        hp.lambda = lambda
        hp.n_hid = n_hid
        hp.reg = reg
        hp.classify = classify
        hp.dropout = dropout
        hp.droplim = droplim
        hp.epochs = epochs
        hp.mb_size_in = mb_size_in
        hp.norm_mode = norm_mode
        hp.opt = opt
        hp.opt_params = opt_params
        hp.learn_decay = learn_decay
        hp.do_batch_norm = do_batch_norm

    # load training data and test data (if any)
    train.inputs, train.targets, test.inputs, test.targets = extract_data(matfname)

    # normalize input data
    if !(norm_mode == "" || lowercase(norm_mode) == "none")
        train.inputs, test.inputs, norm_factors = normalize_inputs(train.inputs, test.inputs, norm_mode)
        nnp.norm_factors = norm_factors   
    end
    # debug
    # println("norm_factors ", typeof(norm_factors))
    # println(norm_factors)

    # set some useful variables
    train.in_k, train.n = size(train.inputs)  # number of features in_k (rows) by no. of examples n (columns)
    train.out_k = size(train.targets,1)  # number of output units
    dotest = size(test.inputs, 1) > 0  # there is testing data -> true, else false

    #  optimization parameters, minibatch, preallocate data storage
    setup_model!(mb, hp, nnp, bn, dotest, train, test)

    # choose layer functions and cost function based on inputs
    setup_functions!(hp.units, train.out_k, hp.opt, hp.classify)

    # statistics for plots and history data
    plotdef = setup_plots(epochs, dotest, plots)

    ##########################################################
    #   neural network training loop
    ##########################################################
    # start the cpu clock
    tic()
    t = 0  # update total counter:  all minibatches and epochs
    # DEBUG
    # println(hp.n_mb)
    # println(hp.mb_size)

    for ep_i = 1:hp.epochs  # loop for "epochs" with counter epoch i as ep_i

        hp.do_learn_decay && step_lrn_decay!(hp, ep_i)

        # Debug
        # ep_i == 3 && error("that's all folks...")

        done = 0
        hp.mb_size = hp.mb_size_in

        # debug
        # println(hp.mb_size)

        for mb_j = 1:hp.n_mb  # loop for mini-batches with counter minibatch j as mb_j
            left = train.n - done
            hp.mb_size = left > hp.mb_size ? hp.mb_size : left
            done += hp.mb_size

            #debug
            # println("minibatch size: $(hp.mb_size)")

            first_example = (mb_j - 1) * hp.mb_size + 1  # mini-batch subset for the inputs->layer 1
            last_example = first_example + hp.mb_size - 1
            colrng = first_example:last_example

            # debug
            # println(colrng)

            t += 1  # update counter

            update_training_views!(mb, train, nnp, hp, colrng)  # next minibatch                

            feedfwd!(mb, nnp, bn,  hp)  # for all layers

            backprop!(nnp, bn, mb, hp, t)  # for all layers

            optimization_function!(nnp, hp, t)

            # update weights, bias, and batch_norm parameters
            @fastmath for hl = 2:nnp.output_layer            
                nnp.theta[hl] .= nnp.theta[hl] .- (hp.alphaovermb .* nnp.delta_w[hl])
                if hp.reg == "L2"  # subtract regularization term
                    nnp.theta[hl] .= nnp.theta[hl] .- (hp.alphaovermb .* (hp.lambda .* nnp.theta[hl]))
                end
                
                if hp.do_batch_norm  # update batch normalization parameters
                    bn.gam[hl][:] -= hp.alphaovermb .* bn.delta_gam[hl]
                    bn.bet[hl][:] -= hp.alphaovermb .* bn.delta_bet[hl]
                else  # update bias
                    nnp.bias[hl] .= nnp.bias[hl] .- (hp.alphaovermb .* nnp.delta_b[hl])
                end

            end  # layer loop

        end # mini-batch loop

        # stats for all mini-batches of one epoch
        gather_stats!(ep_i, plotdef, train, test, nnp, bn, cost_function, train.n, test.n, hp)  

    end # epoch loop

    #####################################################################
    # save, print and plot training statistics after all epochs
    #####################################################################


    # file for simple training stats
    fname = repr(Dates.now())
    fname = "nnstats-" * replace(fname, r"[.:]", "-") * ".txt"
    open(fname, "w") do stats
        println(stats, "Training time: ",toq()," seconds")  # cpu time since tic() =>  toq() returns secs without printing

        feedfwd!(train, nnp, bn, hp, istrain=false)  # output for entire training set
        println(stats, "Fraction correct labels predicted training: ",
                hp.classify == "regression" ? r_squared(train.targets, train.a[nnp.output_layer])
                    : accuracy(train.targets, train.a[nnp.output_layer],epochs))
        println(stats, "Final cost training: ", cost_function(train.targets, train.a[nnp.output_layer], train.n,
                        nnp.theta, hp, nnp.output_layer))

        # output test statistics
        if dotest
            feedfwd!(test, nnp, bn,  hp, istrain=false)
            println(stats, "Fraction correct labels predicted test: ",
                    hp.classify == "regression" ? r_squared(test.targets, test.a[nnp.output_layer])
                        : accuracy(test.targets, test.a[nnp.output_layer],epochs))
            println(stats, "Final cost test: ", cost_function(test.targets, test.a[nnp.output_layer], test.n,
                nnp.theta, hp, nnp.output_layer))
        end

        # output improvement of last 10 iterations for test data
        if plotdef["plot_switch"]["Test"]
            if plotdef["plot_switch"]["Learning"]
                println(stats, "Test data accuracy in final 10 iterations:")
                printdata = plotdef["fracright_history"][end-10+1:end, plotdef["col_test"]]
                for i=1:10
                    @printf(stats, "%0.3f : ", printdata[i])
                end
                print("\n")
            end
        end
    end  # done with stats stream

    # print the stats
    println(read(fname, String))

    # save cost and accuracy from training
    save_plotdef(plotdef)
    
    # plot now
    plot_now && plot_output(plotdef)

    # train inputs, train targets, train predictions, test predictions, trained parameters, batch_norm parms., hyper parms.
    return train.inputs, train.targets, train.a[nnp.output_layer], test.a[nnp.output_layer], nnp, bn, hp;  
     
end  # function run_training


"""
    Create or update views for the training data in minibatches or one big batch
        Arrays: a, z, z_norm, targets  are all fields of struct mb
"""
function update_training_views!(mb::Training_view, train::Model_data, nnp::NN_parameters, 
    hp::Hyper_parameters, colrng::UnitRange{Int64})
    # colrng refers to the set of training examples included in the minibatch
    n_layers = nnp.output_layer
    mb_cols = 1:hp.mb_size  # only reason for this is that the last minibatch might be smaller

    # feedforward:   minibatch views update the underlying data
    mb.a = [view(train.a[i],:,colrng) for i = 1:n_layers]
    mb.z = [view(train.z[i],:,colrng) for i = 1:n_layers]
    mb.targets = view(train.targets,:,colrng)  # only at the output layer

    # training / backprop:  don't need this data and only use minibatch size
    mb.epsilon = [view(train.epsilon[i], :, mb_cols) for i = 1:n_layers]
    mb.grad = [view(train.grad[i], :, mb_cols) for i = 1:n_layers]
    mb.delta_z = [view(train.delta_z[i], :, mb_cols) for i = 1:n_layers]

    if hp.do_batch_norm
        # feedforward
        mb.z_norm = [view(train.z_norm[i],:, colrng) for i = 1:n_layers]
        # backprop
        mb.delta_z_norm = [view(train.delta_z_norm[i], :, mb_cols) for i = 1:n_layers]
    end

    if hp.dropout
        # training:  applied to feedforward, but only for training
        mb.drop_ran_w = [view(train.drop_ran_w[i], :, mb_cols) for i = 1:n_layers]
        mb.drop_filt_w = [view(train.drop_filt_w[i], :, mb_cols) for i = 1:n_layers]
    end

end


# ERROR: MethodError: Cannot `convert` an object of 
# type SubArray{BitArray,2,Array{BitArray,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true} to an object of 
# type SubArray{Bool,2,BitArray{2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true}

"""
function feedfwd!(dat, nnp, bn, do_batch_norm; istrain)
    modifies a, a_wb, z in place to reduce memory allocations
    send it all of the data or a mini-batch

    feed forward from inputs to output layer predictions
"""
function feedfwd!(dat::Union{Model_data, Training_view}, nnp, bn,  hp; istrain=true)

    # hidden layers
    @fastmath for hl = 2:nnp.output_layer-1  
        dat.z[hl][:] = nnp.theta[hl] * dat.a[hl-1]

        if hp.do_batch_norm 
            batch_norm_fwd!(bn, dat, hl, istrain)
            unit_function!(dat.z[hl],dat.a[hl])
        else
            dat.z[hl][:] =  dat.z[hl] .+ nnp.bias[hl]
            unit_function!(dat.z[hl],dat.a[hl])
        end

        if istrain && hp.dropout  
            dropout!(dat,hp,hl)
        end
    end

    # output layer
    @fastmath dat.z[nnp.output_layer][:] = (nnp.theta[nnp.output_layer] * dat.a[nnp.output_layer-1]
        .+ nnp.bias[nnp.output_layer])  # TODO use bias in the output layer with no batch norm? @inbounds 

    classify_function!(dat.z[nnp.output_layer], dat.a[nnp.output_layer])  # a = activations = predictions

end


"""
function backprop!(nnp, dat, do_batch_norm)
    Argument nnp.delta_w holds the computed gradients for weights, delta_b for bias
    Modifies dat.epsilon, nnp.delta_w, nnp.delta_b in place--caller uses nnp.delta_w, nnp.delta_b
    Use for training iterations
    Send it all of the data or a mini-batch
    Intermediate storage of dat.a, dat.z, dat.epsilon, nnp.delta_w, nnp.delta_b reduces memory allocations
"""
function backprop!(nnp, bn, dat, hp, t)

    dat.epsilon[nnp.output_layer][:] = dat.a[nnp.output_layer] .- dat.targets  # @inbounds 
    @fastmath nnp.delta_w[nnp.output_layer][:] = dat.epsilon[nnp.output_layer] * dat.a[nnp.output_layer-1]' # 2nd term is effectively the grad for mse  @inbounds 
    @fastmath nnp.delta_b[nnp.output_layer][:] = sum(dat.epsilon[nnp.output_layer],2)  # @inbounds 

    @fastmath for hl = (nnp.output_layer - 1):-1:2  # loop over hidden layers
        if hp.do_batch_norm
            gradient_function!(dat.z[hl], dat.grad[hl])
            hp.dropout && (dat.grad[hl] .* dat.drop_filt_w[hl])
            dat.epsilon[hl][:] = nnp.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl]  # @inbounds 
            batch_norm_back!(nnp, dat, bn, hl)
            nnp.delta_w[hl][:] = dat.delta_z[hl] * dat.a[hl-1]'  # @inbounds 
        else
            gradient_function!(dat.z[hl], dat.grad[hl])
            hp.dropout && (dat.grad[hl] .* dat.drop_filt_w[hl])
            dat.epsilon[hl][:] = nnp.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl]  # @inbounds 
            nnp.delta_w[hl][:] = dat.epsilon[hl] * dat.a[hl-1]'  # @inbounds 
            nnp.delta_b[hl][:] = sum(dat.epsilon[hl],2)  #  times a column of 1's = sum(row)
        end

    end

end


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


function batch_norm_back!(nnp, dat, bn, hl)
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


"""
    plot_output(plotdef::Dict)

    Plots the plotdef and creates 1 or 2 PyPlot plot windows of the learning (accuracy)
    and/or cost from each training epoch.

"""
function plot_output(plotdef::Dict)
    # plot the progress of training cost and/or learning
    if (plotdef["plot_switch"]["Training"] || plotdef["plot_switch"]["Test"])
        plotlyjs(size=(600,400)) # set chart size defaults

        if plotdef["plot_switch"]["Cost"]
            plt_cost = plot(plotdef["cost_history"], title="Cost Function",
                labels=plotdef["plot_labels"], ylims=(0.0, Inf), bottom_margin=7mm)
            display(plt_cost)  # or can use gui()
        end

        if plotdef["plot_switch"]["Learning"]
            plt_learning = plot(plotdef["fracright_history"], title="Learning Progress",
                labels=plotdef["plot_labels"], ylims=(0.0, 1.05), bottom_margin=7mm) 
            display(plt_learning)
        end

        if (plotdef["plot_switch"]["Cost"] || plotdef["plot_switch"]["Learning"])
            println("Press enter to close plot window..."); readline()
            closeall()
        end
    end
end


function gather_stats!(i, plotdef, train, test, nnp, bn, cost_function, train_n, test_n, hp)

    if plotdef["plot_switch"]["Training"]
        feedfwd!(train, nnp, bn, hp, istrain=false)

        if plotdef["plot_switch"]["Cost"]
            plotdef["cost_history"][i, plotdef["col_train"]] = cost_function(train.targets,
                train.a[nnp.output_layer], train_n, nnp.theta, hp, nnp.output_layer)
        end
        if plotdef["plot_switch"]["Learning"]
            plotdef["fracright_history"][i, plotdef["col_train"]] = (  hp.classify == "regression"
                    ? r_squared(train.targets, train.a[nnp.output_layer])
                    : accuracy(train.targets, train.a[nnp.output_layer], i)  )
        end
    end

    if plotdef["plot_switch"]["Test"]
        feedfwd!(test, nnp, bn, hp, istrain=false)

        if plotdef["plot_switch"]["Cost"]
            cost = cost_function(test.targets,
                test.a[nnp.output_layer], test.n, nnp.theta, hp, nnp.output_layer)
                # println("iter: ", i, " ", "cost: ", cost)
            plotdef["cost_history"][i, plotdef["col_test"]] =cost
        end
        if plotdef["plot_switch"]["Learning"]
            # printdims(Dict("test.a"=>test.a, "test.z"=>test.z))
            plotdef["fracright_history"][i, plotdef["col_test"]] = (  hp.classify == "regression"
                    ? r_squared(test.targets, test.a[nnp.output_layer])
                    : accuracy(test.targets, test.a[nnp.output_layer], i)  )
        end
    end

    # println("train 1 - predictions:")
    # println(1.0 .- train.a[nnp.output_layer][:, 1:2])

    # println("test 1 - predictions:")
    # println(1.0 .- test.a[nnp.output_layer][:, 1:2])
    
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


function wrong_preds(targets, preds, cf = !isequal)
    if size(targets,1) > 1
        targetmax = ind2sub(size(targets),vec(findmax(targets,1)[2]))[1]
        predmax = ind2sub(size(preds),vec(findmax(preds,1)[2]))[1]
        wrongs = find(cf.(targetmax, predmax))
    else
        # works because single output unit is sigmoid--well, what do we do about regression? we use r_squared
        choices = [j >= 0.5 ? 1.0 : 0.0 for j in preds]
        wrongs = find(cf.(choices, targets))
    end
    return wrongs
end


function right_preds(targets, preds)
    return wrong_preds(targets, preds, isequal)
end

function r_squared(targets, preds)
    ybar = mean(targets)
    return 1.0 - sum((targets .- preds).^2.) / sum((targets .- ybar).^2.)
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

    function predict()

Two methods:
    with a .mat file as input: 
        function predict(matfname::String, hp, nnp, bn; test::Bool=false, norm_mode::String="")
    with arrays as input:
        function predict(inputs, targets, hp, nnp, bn, norm_factors)

Generate predictions given previously trained parameters and input data.
Not suitable in a loop because of all the additional allocations.
Use with one-off needs like scoring a test data set or
producing predictions for operational data fed into an existing model.
"""
function nnpredict(matfname::String, hp, nnp, bn; test::Bool=false)
    if !test
        inputs, targets, _, __ = extract_data(matfname)  # training data
    else
        _, __, inputs, targets = extract_data(matfname)  # test data
    end

    nnpredict(inputs, targets, hp, nnp, bn)
end


function nnpredict(inputs, targets, hp, nnp, bn)
    dataset = Model_data()
        dataset.inputs = inputs
        dataset.targets = targets
        dataset.in_k, dataset.n = size(inputs)  # number of features in_k (rows) by no. of examples n (columns)
        dataset.out_k = size(dataset.targets,1)  # number of output units

    if hp.norm_mode == "standard" || hp.norm_mode == "minmax"
        normalize_replay!(dataset.inputs, hp.norm_mode, nnp.norm_factors)
    end

    preallocate_data!(dataset, nnp, dataset.n, hp)

    setup_functions!(hp.units, dataset.out_k, hp.opt, hp.classify)  # for feedforward calculations

    feedfwd!(dataset, nnp, bn, hp, istrain=false)  # output for entire dataset

    println("Fraction correct labels predicted: ",
        hp.classify == "regression" ? r_squared(dataset.targets, dataset.a[nnp.output_layer])
                                    : accuracy(dataset.targets, dataset.a[nnp.output_layer], hp.epochs))
    return dataset.a[tp.output_layer]
end

end  # module GeneralNN
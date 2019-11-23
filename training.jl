
"""
function setup_training(epochs::Int64, hidden::Array{Tuple{String,Int64},1}; alpha::Float64=0.35
    mb_size_in::Int64=0, lambda::Float64=0.01, classify::String="softmax", norm_mode::String="none"
    opt::String="", opt_params::Array{Float64,1}=[0.9,0.999], do_batch_norm::Bool=false
    reg::String="L2", dropout::Bool=false, droplim::Array{Float64,1}=[0.5], plots::Array{String,1}=["Training", "Learning"]
    learn_decay::Array{Float64,1}=[1.0, 1.0])

Train sigmoid/softmax neural networks up to 11 layers.  Detects
number of output labels from data. Detects number of features from data for input units.
Enables any size minibatch (last batch will be smaller if minibatch size doesn't divide evenly
into number of examples).  Plots learning and cost outcomes by epoch for training and test data--or by
minibatch, but this is VERY slow.

This is a front-end function that verifies all inputs and calls _run_training(). A convenience method allows 
all of the input parameters to be read from a TOML file.  This is further explained below.


    returns a dict containing these keys:  
        train_inputs      ::= after any normalization
        train_targets     ::= matches original inputs
        train_preds       ::= using final values of trained parameters
        wgts         ::= struct that holds all trained parameters
        batch_norm_params ::= struct that holds all batch_norm parameters
        hyper_params      ::= all hyper parameters used to control training

    and if test/validation data is present, these additional keys:
        test_inputs       ::= after any normalization
        test_targets      ::= matches original inputs
        test_preds        ::= using final values of trained parameters

    required inputs:
        datalist
        epochs
        hidden

    optional named parameters:
        alpha           ::= learning rate
        lambda          ::= regularization rate
        dobatch         ::=Bool, true to use minibatch training
        mb_size_in      ::= minibatch size=>integer
        hidden          ::= array of tuple(Sting,Int) containing type and number of units in each hidden layer;
                            make sure to use an array even with 1 hidden layer as in [("relu",40)];
                            use [0] to indicate no hidden layer (typically for linear regression)
        norm_mode       ::= "standard", "minmax" or "" => normalize inputs
        do_batch_norm   ::= true or false => normalize each hidden layers linear outputs
        opt             ::= one of "Momentum", "RMSProp", "Adam" or "".  default is blank string "".
        opt_params      ::= parameters used by Momentum, RMSProp, or Adam
                           Momentum, RMSProp: one floating point value as [.9] (showing default)
                           Adam: 2 floating point values as [.9, .999] (showing defaults)
                           Note that epsilon is ALWAYS set to 1e-8
                           To accept defaults, don't input this parameter or use []
        classify        ::= "softmax", "sigmoid", "logistic", or "regression" for only the output layer
        stats           ::= determines training statistics collected and plotted
                            any choice of ["learning", "cost", "train", "test"];
                            for no stats use [""] or ["none"]
                            include "batch" to collect stats on each minibatch and/or "epoch" (the default) for stats per epoch
                            warning:  stats for every batch is SLOW
        reg             ::= type of regularization, must be one of "L1", "L2", "Maxnorm", ""
        maxnorm_lim     ::= array of limits set for hidden layers + output layer
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
        plot_now        ::Bool.  If true, plot training stats immediately and save the statsdat that contains stats 
                            gathered while running the training.  You can plot the file separately, later.
        sparse          ::Bool. If true, input data will be treated and maintained as SparseArrays.
        initializer     ::= "xavier", "uniform", "normal" or "zero" used to set how Wgts, not including bias, are initialized.
        scale_init      ::= Float64. Scale the initialization values for the parameters.
        bias_initializer::= Float64. 0.0, 1.0 or float inbetween. Initial value for bias terms.
        quiet           ::Bool. true is default to suppress progress messages during preparation and training.

The following method allows all input parameters to be supplied by a TOML file:

    function train_nn(datalist, argsfile::String, errorcheck::Bool=false)

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
        opt_params = [0.9, 0.999]  
        learn_decay = [0.5,4.0]
        dobatch = true
        do_batch_norm =  true
        sparse = false
        initializer = "xavier"          # or "normal", "uniform"
        scale_init = 2.0, 
        bias_initializer  = 0.0
        quiet = true
        stats = ["Train", "Learning", "Test", "epoch", "batch"]           
        plot_now = true                    

        The last 4 items are not training hyperparameters, but control plotting and the process.

    If errorcheck is set to true the TOML file is checked:
       1) To make sure all required arguments are present; this is true even
          though the function that will be called provides valid defaults.
       2) To make sure that all argument names are valid.
    If any errors are found, neural network training is not run.
"""
function setup_training(       
                epochs::Int64, 
                hidden::Array{Tuple{String,Int64},1}=[("none",0)];   # required
                alpha::Float64=0.35,                              # named, optional if defaults ok
                mb_size_in::Int64=0, 
                lambda::Float64=0.01, 
                classify::String="softmax", 
                norm_mode::String="none", 
                opt::String="", 
                opt_params::Array{Float64,1}=[0.9,0.999], 
                dobatch=false, 
                do_batch_norm::Bool=false, 
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

    argsdict = Dict(
        "epochs"            => epochs,
        "hidden"            => hidden,
        "alpha"             =>  alpha,        
        "mb_size_in"        =>  mb_size_in, 
        "lambda"            =>  lambda, 
        "classify"          =>  classify, 
        "norm_mode"         =>  norm_mode, 
        "opt"               =>  opt, 
        "opt_params"        =>  opt_params, 
        "dobatch"           =>  dobatch, 
        "do_batch_norm"     =>  do_batch_norm, 
        "reg"               =>  reg, 
        "maxnorm_lim"       =>  maxnorm_lim, 
        "dropout"           =>  dropout, 
        "droplim"           =>  droplim, 
        "stats"             =>  stats, 
        "learn_decay"       =>  learn_decay, 
        "plot_now"          =>  plot_now, 
        "sparse"            =>  sparse, 
        "initializer"       =>  initializer, 
        "scale_init"        =>  scale_init, 
        "bias_initializer"  =>  bias_initializer, 
        "quiet"             =>  quiet  
        )

    args_verify(argsdict)  # no return; errors out at first error

    hp = build_hyper_parameters(argsdict)

    # validate_datafiles(datalist) # no return; errors out if errors
 
    return hp
end


function setup_training(argsfile::String)
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

    # validate_datafiles(datalist) # no return; errors out if errors

    return hp

end


function pretrain(dat_x, dat_y, hp)
    Random.seed!(70653)  # seed int value is meaningless

    # 1. instantiate data containers
        dat = Model_data()
        mb = Batch_view()
        nnw = Wgts()
        bn = Batch_norm_params()
        dat.inputs, dat.targets = dat_x, dat_y
        dat.in_k, dat.n = size(dat_x)
        dat.out_k = size(dat_y, 1)

    # 2. optimization parameters, minibatches, regularization
        prep_training!(mb, hp, nnw, bn, dat.n)

    # 3. normalize data
        if !(hp.norm_mode == "" || lowercase(hp.norm_mode) == "none")
            nnw.norm_factors = normalize_inputs!(dat.inputs, hp.norm_mode)
        end       

    # 4. preallocate model structure for weights and minibatch 
        !hp.quiet && println("Pre-allocate weights and minibatch storage starting")
        preallocate_wgts!(nnw, hp, dat.in_k, dat.n, dat.out_k)
        hp.dobatch && preallocate_minibatch!(mb, nnw, hp) 
        hp.do_batch_norm && preallocate_batchnorm!(bn, mb, nnw.ks)
        !hp.quiet && println("Pre-allocate weights and minibatch storage completed")

    # 5. choose layer functions and cost function based on inputs
        setup_functions!(hp, nnw, bn, dat) 

    # 6. preallocate storage for data transforms
        preallocate_data!(dat, nnw, dat.n, hp)

    return dat, mb, nnw, bn
    !hp.quiet && println("Training setup complete")
end

function prepredict(dat_x, dat_y, hp, nnw; notrain=true)
    notrain && (Random.seed!(70653))  # seed int value is meaningless

    # 1. instantiate data containers  
    !hp.quiet && println("Instantiate data containers")

        dat = Model_data()   # for test--but there is no training, just prediction
        # bn = Batch_norm_params()  # not used, but needed for API

        !hp.quiet && println("Set input data aliases to model data structures")
        dat.inputs, dat.targets = dat_x, dat_y
        !hp.quiet && println("Alias to model data structures completed")

        # set some useful variables
        dat.in_k, dat.n = size(dat_x)  # number of features in_k (rows) by no. of examples n (columns)
        dat.out_k = size(dat_y,1)  # number of output units

    # 2. not needed
    # 3. normalize data
        if !(hp.norm_mode == "" || lowercase(hp.norm_mode) == "none")
            nnw.norm_factors = normalize_inputs!(dat.inputs, hp.norm_mode)
        end       

    # 4. not needed

    # 5. choose layer functions and cost function based on inputs
        notrain && (setup_functions!(hp, nnw, bn, dat)) 

    # 6. preallocate storage for data transforms
        !hp.quiet && println("Pre-allocate storage starting")
        # preallocate_wgts!(nnw, hp, train.in_k, train.n, train.out_k)
        preallocate_data!(dat, nnw, dat.n, hp)
        # hp.dobatch && preallocate_minibatch!(mb, nnw, hp) 
        # hp.do_batch_norm && preallocate_batchnorm!(bn, mb, nnw.ks)
        !hp.quiet && println("Pre-allocate storage completed")

    return dat
end


"""
function train(datalist, hp)

This is the one function and only method that really does the work.  It runs
the model training or as many frameworks refer to it, "fits" the model.
"""
function train(train_x, train_y, hp, testgrad=false)
    !hp.quiet && println("Training setup beginning")

    train, mb, nnw, bn = pretrain(train_x, train_y, hp)
    statsdat = setup_stats(hp, false)  # GET RID of dotest here

    !hp.quiet && println("Training setup complete")

    training_time = training_loop!(hp, train, mb, nnw, bn, statsdat)

    # save, print and plot training statistics
    output_stats(train, nnw, bn, hp, training_time, statsdat)

    ret = Dict(
                "train_inputs" => train_x, 
                "train_targets"=> train_y, 
                "train_preds" => train.a[nnw.output_layer], 
                "Wgts" => nnw, 
                "batchnorm_params" => bn, 
                "hyper_params" => hp
                )

    return ret

end # _run_training_core, method with test data


function train(train_x, train_y, test_x, test_y, hp, testgrad=false)
    !hp.quiet && println("Training setup beginning")

    train, mb, nnw, bn = pretrain(train_x, train_y, hp)
    test = prepredict(test_x, test_y, hp, nnw, notrain=false)
        # use notrain=false because the test data is used during training
    statsdat = setup_stats(hp, true)  # GET RID of dotest here

    !hp.quiet && println("Training setup complete")

    training_time = training_loop!(hp, train, test, mb, nnw, bn, statsdat)

    # save, print and plot training statistics
    output_stats(train, test, nnw, bn, hp, training_time, statsdat)

    ret = Dict(
                "train_inputs" => train_x, 
                "train_targets"=> train_y, 
                "train_preds" => train.a[nnw.output_layer], 
                "Wgts" => nnw, 
                "batchnorm_params" => bn, 
                "hyper_params" => hp,
                "test_inputs" => test.inputs, 
                "test_targets" => test.targets, 
                "test_preds" => test.a[nnw.output_layer]  
                )


    
    return ret

end # _run_training_core, method with test data





# function train(datalist, hp, testgrad=false)

#     !hp.quiet && println("Training setup beginning")
#     !hp.quiet && println("dobatch: ", hp.dobatch)
#     # seed random number generator.  For runs of identical models the same weight initialization
#     # will be used, given the number of parameters to be estimated.  Enables better comparisons.
#     Random.seed!(70653)  # seed int value is meaningless

#     if size(datalist,1) == 4
#         train_x = datalist[1]
#         train_y = datalist[2]
#         test_x = datalist[3]
#         test_y = datalist[4]
#         dotest = size(test_x, 1) > 0  # there is testing data -> true, else false
#         if !dotest
#             error("Test data inputs are empty. Rerun without passing test data inputs at all.")
#         end
#     elseif size(datalist, 1) == 2
#         train_x = datalist[1]
#         train_y = datalist[2]
#         dotest = false
#     else
#         error("Datalist input contained wrong number of inputs")
#     end

#     ##################################################################################
#     #   setup model: data structs, many control parameters, functions,  memory pre-allocation
#     #################################################################################

#     # 1. instantiate data containers  
#     !hp.quiet && println("Instantiate data containers")

#         train = Model_data()  # train holds all the training data and layer inputs/outputs
#         dotest && (test = Model_data())   # for test--but there is no training, just prediction
#         mb = Batch_view()  # even if not used, initialization is empty
#         nnw = Wgts()  # neural network trained parameters
#         bn = Batch_norm_params()  # do we always need the data structure to run?  yes--TODO fix this

#         !hp.quiet && println("Set input data aliases to model data structures")
#         if dotest
#             train.inputs, train.targets, test.inputs, test.targets = train_x, train_y, test_x, test_y
#         else
#             train.inputs, train.targets = train_x, train_y
#         end
#         !hp.quiet && println("Alias to model data structures completed")

#         # set some useful variables
#         train.in_k, train.n = size(train_x)  # number of features in_k (rows) by no. of examples n (columns)
#         dotest && ((test.in_k, test.n) = size(test_x))
#         train.out_k = size(train_y,1)  # number of output units

#     #  2. optimization parameters, minibatches, regularization
#         prep_training!(mb, hp, nnw, bn, train)

#     # 3. normalize data  
#         if !(hp.norm_mode == "" || lowercase(hp.norm_mode) == "none")
#             nnw.norm_factors = normalize_inputs!(train.inputs, hp.norm_mode)
#             dotest && normalize_inputs!(test.inputs, nnw.norm_factors, hp.norm_mode) 
#         end

#     # 4. preallocate data storage  
#         !hp.quiet && println("Pre-allocate storage starting")
#         preallocate_wgts!(nnw, hp, train.in_k, train.n, train.out_k)
#         preallocate_data!(train, nnw, train.n, hp)
#         hp.dobatch && preallocate_minibatch!(mb, nnw, hp) 
#         hp.do_batch_norm && preallocate_batchnorm!(bn, mb, nnw.ks)
#         dotest && preallocate_data!(test, nnw, test.n, hp)
#         !hp.quiet && println("Pre-allocate storage completed")

#     # 5. choose layer functions and cost function based on inputs
#         setup_functions!(hp, nnw, train) 

#     # 6. statistics for plots and history data
#         statsdat = setup_stats(hp, dotest)  # TRAIN

#     !hp.quiet && println("Training setup complete")
    


#     ##########################################################
#     #   neural network training loop
#     ##########################################################
#     datalist = dotest ? [train, test] : [train]

#     ##########################################################
#     #   test gradients using the defined model and  STOP
#     ##########################################################
#         # test gradients against numerical gradients
#         # we need to further factor training to capture everything it sets up
#         # as output and have another outer function that then runs the training_loop

#         # testgrad && begin
#         #    verify_gradient(train, nnw, bn, hp)
#         #    return
#         # end

#     testgrad && begin 
#         check_grads(hp) 
#         return
#     end
    
#     training_time = training_loop!(hp, datalist, mb, nnw, bn, statsdat)

    
#     # save, print and plot training statistics
#     output_stats(datalist, nnw, bn, hp, training_time, statsdat)

#     ret = Dict(
#                 "train_inputs" => train_x, 
#                 "train_targets"=> train_y, 
#                 "train_preds" => train.a[nnw.output_layer], 
#                 "Wgts" => nnw, 
#                 "batchnorm_params" => bn, 
#                 "hyper_params" => hp
#                 )

#     dotest &&   begin
#                     ret["test_inputs"] = test.inputs 
#                     ret["test_targets"] = test.targets 
#                     ret["test_preds"] = test.a[nnw.output_layer]  
#                 end 

#     return ret

# end # _run_training_core, method with test data


# TODO do we need to compare the test and train features, labels
function validate_datafiles(dat_x, dat_y)
   # if size(datalist,1) == 4
   #      train_x = datalist[1]
   #      train_y = datalist[2]
   #      test_x = datalist[3]
   #      test_y = datalist[4]
   #      dotest = size(test_x, 1) > 0  # there is testing data -> true, else false
   #  elseif size(datalist, 1) == 2
   #      train_x = datalist[1]
   #      train_y = datalist[2]
   #      dotest = false
   #  else
   #      error("Datalist input contained wrong number of input arrays")
   #  end
    
    # training data
    (dat_m, dat_n) = size(dat_x)
    (target_m, target_n) = size(dat_y)
    if dat_m >= dat_n
        @warn("No. of features is greater than no. of samples. Maybe the training array must be tranposed.")
    end
    if target_m >= target_n
        error("No. of output labels is greater than no. of samples. Probably the label array must be transposed.")
    end
    if target_n != dat_n
        error("No. of training inputs does not match no. of training label outputs.")
    end
    
    # # test or validation data
    # if dotest
    #     (test_m, test_n) = size(test_x)
    #     (testy_m, testy_n) = size(test_y)
    #     if test_m >= test_n
    #         error("No. of features is greater than no. of samples. Probably the test array must be tranposed.")
    #     end
    #     if testy_m >= testy_n
    #         error("No. of test output labels is greater than no. of samples. Probably the test label array must be transposed.")
    #     end
    #     if testy_n != test_n
    #         error("No. of test inputs does not match no. of test label outputs.")
    #     end
        
    #     if train_m != test_m
    #         error("No. of training features does not match test features.")
    #     end   
    # end
end


"""
    function train(train_x, train_y, hp, testgrad=false)
    function train(train_x, train_y, test_x, test_y, hp, test_grad=false)

Train sigmoid/softmax neural networks up to 11 layers.  Detects
number of output labels from data. Detects number of features from data for input units.
Enables any size minibatch (last batch will be smaller if minibatch size doesn't divide evenly
into number of examples).  Plots learning and cost outcomes by epoch for training and test data--or by
minibatch, but this is VERY slow.

This is the one function that really does the work.  It runs
the model training or as many frameworks refer to it, "fits" the model.

Train logistic regression, linear regression and sigmoid/softmax neural networks, up to 11 layers.  Detects
number of output labels from data. Detects number of features from data for input units.
Enables any size minibatch (last batch will be smaller if minibatch size doesn't divide evenly
into number of examples).  Plots learning and cost outcomes by epoch for training and test data--or by
minibatch, but this is VERY slow.

First, run setup_params using either a TOML file as input or named arguments.  See ?setup_params!  
Second, prepare you data files with examples as COLUMNS and input features as rows.
With minibatch training, you should shuffle your training data.  You may use shuffle_data!(train_x, train_y).
Now, you are ready to run function train.

    returns a dict containing these keys:  
        train_inputs      ::= after any normalization
        train_targets     ::= matches original inputs
        train_preds       ::= using final values of trained parameters
        wgts              ::= struct that holds all trained parameters
        batch_norm_params ::= struct that holds all batch_norm parameters
        hyper_params      ::= all hyper parameters used to control training
        stats             ::= dict of training statistics by epoch or batch, including 
                              learning (naive accuracy) and cost for training and test 
                              data, based on your inputs to hp.

    and if test/validation data is present, these additional keys:
        test_inputs       ::= after any normalization
        test_targets      ::= matches original inputs
        test_preds        ::= using final values of trained parameters
"""
function train(train_x, train_y, hp; testgrad=false, ret="basic")
# method with no test data--stub in zeros
    test_x = zeros(0,0); test_y = zeros(0,0)

    ret = _train(train_x, train_y, test_x, test_y, hp; testgrad=testgrad, ret=ret)

    return ret

end 


function train(train_x, train_y, test_x, test_y, hp; testgrad=false, ret="basic")
# method that includes test data

    ret = _train(train_x, train_y, test_x, test_y, hp; testgrad=testgrad, ret=ret)

    return ret

end 


function _train(train_x, train_y, test_x, test_y, hp; testgrad=false, ret="basic")
    train, mb, nnw, bn, model = pretrain(train_x, train_y, hp)
    test = prepredict(test_x, test_y, hp, nnw, notrain=false) #  notrain=false because test data used during training
    dotest = isempty(test.inputs) ? false : true
    stats = setup_stats(hp, dotest)  

    if hp.dobatch
        train_method = minibatch_training
    else
        train_method = fullbatch_training
    end

    training_time = training_loop!(hp, train, test, mb, nnw, bn, stats, model, train_method)

    # save, print and plot training statistics
    output_stats(train, test, nnw, hp, bn, training_time, stats, model)

    if ret == "basic"  
            out = Dict(
                    # "train_inputs" => train_x, 
                    # "train_targets"=> train_y, 
                    # "train_preds" => train.a[nnw.output_layer], 
                    "wgts" => nnw, 
                    "batchnorm_params" => bn, 
                    "hyper_params" => hp,
                    # "stats" => stats,
                    "model" => model
                    )
    elseif ret == "all"
            out = Dict(
                    "train_inputs" => train_x, 
                    "train_targets"=> train_y, 
                    "train_preds" => train.a[nnw.output_layer], 
                    "wgts" => nnw, 
                    "batchnorm_params" => bn, 
                    "hyper_params" => hp,
                    "stats" => stats,
                    "model" => model
                    )        
        if dotest
            try
                out["test_inputs"] = test.inputs 
                out["test_targets"] = test.targets 
                out["test_preds"] = test.a[nnw.output_layer]
            catch
            end
        end
    end

    return out

end


function pretrain(dat_x, dat_y, hp)
    Random.seed!(70653)  # seed int value is meaningless

    # 1. instantiate data and model containers
        dat = Model_data()
        model = Model_def()
        mb = Batch_view()
        nnw = Wgts()
        bn = Batch_norm_params()
        dat.inputs, dat.targets = dat_x, dat_y
        dat.in_k, dat.n = size(dat_x)
        dat.out_k = size(dat_y, 1)

    # 2. setup parameters for batch_size, learn_decay, dropout, and Maxnorm regularization
        prep_training!(hp, dat.n)

    # 3. normalize data
        if !(hp.norm_mode == "" || lowercase(hp.norm_mode) == "none")
            nnw.norm_factors = normalize_inputs!(dat.inputs, hp.norm_mode)
        end       

    # 4. preallocate model structure for weights and minibatch 
        !hp.quiet && println("Pre-allocate weights and minibatch storage starting")
        preallocate_wgts!(nnw, hp, dat.in_k, dat.n, dat.out_k)
        hp.dobatch && preallocate_minibatch!(mb, nnw, hp) 
        hp.do_batch_norm && preallocate_bn_params!(bn, mb, nnw.ks)
        !hp.quiet && println("Pre-allocate weights and minibatch storage completed")

    # 5. choose layer functions and cost function based on inputs => all in model
        create_model!(model, hp, dat.out_k)

    # 6. preallocate storage for data transforms
        preallocate_data!(dat, nnw, dat.n, hp)

    return dat, mb, nnw, bn, model
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
        out_k = dat.out_k = size(dat_y,1)  # number of output units

    # 2. optimization parameters, minibatches, regularization
        # not needed
        
    # 3. normalize data
        if !(hp.norm_mode == "" || lowercase(hp.norm_mode) == "none")
            nnw.norm_factors = normalize_inputs!(dat.inputs, nnw.norm_factors, hp.norm_mode)
        end       

    # 4. preallocate model structure for weights and minibatch 
        # not needed

    # 5. choose layer functions and cost function based on inputs
        # not neededs

    # 6. preallocate storage for data transforms
        !hp.quiet && println("Pre-allocate storage starting")
        # preallocate_wgts!(nnw, hp, train.in_k, train.n, train.out_k)
        preallocate_data!(dat, nnw, dat.n, hp)
        # hp.dobatch && preallocate_minibatch!(mb, nnw, hp) 
        # hp.do_batch_norm && preallocate_bn_params!(bn, mb, nnw.ks)
        !hp.quiet && println("Pre-allocate storage completed")

    return dat
end



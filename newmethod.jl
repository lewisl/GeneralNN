function train(train_x, train_y, hp, testgrad=false)
    !hp.quiet && println("Training setup beginning")

    train, mb = pretrain(train_x, train_y, hp)
    stats = setup_stats(hp, false)  # GET RID of dotest here

    !hp.quiet && println("Training setup complete")

    training_time = training_loop!(hp, train, mb, nnw, bn, stats)

    # save, print and plot training statistics
    output_stats(train, nnw, bn, hp, training_time, stats)

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

    train, mb, nnw = pre_train(train_x, train_y, hp)
    test = pre_predict(test_x, test_y, hp, nnw)
    stats = setup_stats(hp, true)  # GET RID of dotest here

    !hp.quiet && println("Training setup complete")

    training_time = training_loop!(hp, train, test, mb, nnw, bn, stats)

    # save, print and plot training statistics
    output_stats(train, test, nnw, bn, hp, training_time, stats)

    ret = Dict(
                "train_inputs" => train_x, 
                "train_targets"=> train_y, 
                "train_preds" => train.a[nnw.output_layer], 
                "Wgts" => nnw, 
                "batchnorm_params" => bn, 
                "hyper_params" => hp
                )

        ret["test_inputs"] = test.inputs 
        ret["test_targets"] = test.targets 
        ret["test_preds"] = test.a[nnw.output_layer]  
    
    return ret

end # _run_training_core, method with test data


function pre_train(dat_x, dat_y, hp)
    Random.seed!(70653)  # seed int value is meaningless

    # 1. instantiate data containers
        dat = Model_data()
        mb = Batch_view()
        nnw = Wgts()
        bn = Batch_norm_params()
        dat.inputs, dat.targets = dat_x, dat_y
        dat.in_k, dat.n = size(train_x)
        dat.out_k = size(train_y, 1)

    # 2. optimization parameters, minibatches, regularization
        prep_training!(mb, hp, nnw, bn, dat.n)

    # 3. normalize data
        if !(hp.norm_mode == "" || lowercase(hp.norm_mode) == "none")
            nnw.norm_factors = normalize_inputs!(train.inputs, hp.norm_mode)
        end       

    # 4. preallocate model structure for weights and minibatch 
        !hp.quiet && println("Pre-allocate weights and minibatch storage starting")
        preallocate_wgts!(nnw, hp, dat.in_k, dat.n, dat.out_k)
        hp.dobatch && preallocate_minibatch!(mb, nnw, hp) 
        hp.do_batch_norm && preallocate_bn_params!(bn, mb, nnw.ks)
        !hp.quiet && println("Pre-allocate weights and minibatch storage completed")

    # 5. choose layer functions and cost function based on inputs
        setup_functions!(hp, bn, nnw, train) 

    # 6. preallocate storage for data transforms
        preallocate_data!(dat, nnw, dat.n, hp)

    return dat, mb, nnw
    !hp.quiet && println("Training setup complete")
end

function pre_predict(dat_x, dat_y, hp, nnw; notrain=true)
    notrain && (Random.seed!(70653))  # seed int value is meaningless

    # 1. instantiate data containers  
    !hp.quiet && println("Instantiate data containers")

        dat = Model_data())   # for test--but there is no training, just prediction

        !hp.quiet && println("Set input data aliases to model data structures")
        dat.inputs, dat.targets = dat_x, dat_y
        !hp.quiet && println("Alias to model data structures completed")

        # set some useful variables
        dat.in_k, dat.n = size(dat_x)  # number of features in_k (rows) by no. of examples n (columns)
        dat.out_k = size(dat_y,1)  # number of output units

    # 2. not needed
    # 3. normalize data
        normalize_inputs!(dat.inputs, nnw.norm_factors, hp.norm_mode) 

    # 4. not needed

    # 5. choose layer functions and cost function based on inputs
        notrain && (setup_functions!(hp, bn, nnw, train)) 

    # 6. preallocate storage for data transforms
        !hp.quiet && println("Pre-allocate storage starting")
        # preallocate_wgts!(nnw, hp, train.in_k, train.n, train.out_k)
        preallocate_data!(dat, nnw, dat.n, hp)
        # hp.dobatch && preallocate_minibatch!(mb, nnw, hp) 
        # hp.do_batch_norm && preallocate_bn_params!(bn, mb, nnw.ks)
        !hp.quiet && println("Pre-allocate storage completed")

    return dat
end
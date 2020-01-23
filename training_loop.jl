
using StatsBase # basic statistical functions

# method with no input test data
function training_loop!(hp, train, mb, nnw, bn, stats, model)
    _training_loop!(hp, train, Model_data(), mb, nnw, bn, stats, model)
        # Model_data() passes an empty test data object
end

# method with input of train and test data: test data used for training stats
"""
    training_loop!(hp, train, mb, nnw, bn, stats)
    training_loop!(hp, train, test, mb, nnw, bn, stats)

    Inputs:
    hp:       Hyper_parameters object
    train:    Model_data object
    mb:       Batch_view object (for minibatches--provide one even 
                   if it is empty and unused)
    nnw:      Wgts (trained parameters) object
    bn:       Batch_norm_params object
    stats:    Dict created by setup_stats function to hold training statistics

Performs machine learning training using gradient descent. Enables minibatch learning and stochastic gradient
descent with a batch size of 1. The full loop includes feed forward, back propagation, optimization of 
parameter updates and updating the trained parameters. The first method does not include test data. The second method
includes the test Model_data object to track training statistics on how cost and accuracy change for the 
test or validation data set.
"""
function training_loop!(hp, train, test, mb, nnw, bn, stats, model)
    _training_loop!(hp, train, test, mb, nnw, bn, stats, model)
end


function _training_loop!(hp, train, test, mb, nnw, bn, stats, model)
!hp.quiet && println("training_loop(hp, train, test mb, nnw, bn, stats; dotest=false)")

    dotest = isempty(test.inputs) ? false : true

    # println("Experiment with eval")
    # x = mini_eval(train)
    # println("Before: ", train.a[2][1])
    # hl = 2; 
    # dat = train
    # println(x[1], " ", x[2])
    # pair(x[1], x[2])
    # println("After: ", train.a[2][1])

    training_time = @elapsed begin # start the cpu clock and begin block for training process
        t = 0  # counter:  number of times parameters will have been updated: minibatches * epochs

        for ep_i = 1:hp.epochs  # loop for "epochs" with counter epoch i as ep_i
            !hp.quiet && println("Start epoch $ep_i")
            hp.do_learn_decay && step_learn_decay!(hp, ep_i)

            if hp.dobatch  # minibatch training

                for colrng in MBrng(train.n, hp.mb_size_in)  # set setup.jl for definition of iterator
                    hp.mb_size = mbsize(colrng)
  
                    !hp.quiet && println("   Start minibatch for ", colrng)           
                    
                    update_batch_views!(mb, train, nnw, hp, colrng)  # select data columns for the minibatch   

                    t += 1   # number of executions of minibatch loop
                    train_one_step!(mb, nnw, bn, hp, t, model)

                    # stats for each minibatch--expensive!!!
                    stats["period"] == "batch" && begin
                        gather_stats!(stats, "train", t, train, nnw, cost_function, hp, bn, model.ff_execstack)  
                        dotest && gather_stats!(stats, "test", t, test, nnw, cost_function, hp, bn, model.ff_execstack) 
                    end

                end # mini-batch loop

            else
                t += 1
                train_one_step!(train, nnw, bn, hp, t, model)
            end

            # stats across all mini-batches of one epoch (e.g.--no stats per minibatch)
            stats["period"] == "epoch" && begin
                gather_stats!(stats, "train", ep_i, train, nnw, cost_function, hp, bn, model.ff_execstack)  
                dotest && gather_stats!(stats, "test", ep_i, test, nnw, cost_function, hp, bn, model.ff_execstack) 
            end

        end # epoch loop
    end # training_time begin block

    return training_time
end # function training_loop

# function train_one_step!(dat, nnw, bn, hp, t)
function train_one_step!(dat, nnw, bn, hp, t, model)

    feedfwd!(dat, nnw, hp, bn, model.ff_execstack)  # for all layers
    backprop!(nnw, dat, hp)  # for all layers   
    optimization_function!(nnw, hp, bn, t)
    update_parameters!(nnw, hp, bn)

end


# little helper function
mbsize(colrng) = float(size(colrng, 1))

#########################################################
#  functions inside the training loop
#########################################################


"""
function feedfwd!(dat, nnw, do_batch_norm)
    modifies a, a_wb, z in place to reduce memory allocations
    send it all of the data or a mini-batch

    feed forward from inputs to output layer predictions
"""
function feedfwd!(dat::Union{Batch_view,Model_data}, nnw, hp, bn, ff_execstack)  
!hp.quiet && println("feedfwd!(dat::Union{Batch_view, Model_data}, nnw, hp)")

    # # dropout for input layer (if probability < 1.0) or noop
    # dropout_fwd_function![1](dat,hp,1)  

    # # hidden layers
    # @fastmath @inbounds for hl = 2:nnw.output_layer-1  
    #     affine_function!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl]) # if do_batch_norm, ignores bias arg
    #     batch_norm_fwd_function!(dat, hl)  # do it or noop
    #     unit_function![hl](dat.a[hl], dat.z[hl]) # per setup_functions
    #     dropout_fwd_function![hl](dat,hp,hl)  # do it or noop
    # end

    # # output layer
    # @inbounds affine!(dat.z[nnw.output_layer], dat.a[nnw.output_layer-1], 
    #                   nnw.theta[nnw.output_layer], nnw.bias[nnw.output_layer])
    # classify_function!(dat.a[nnw.output_layer], dat.z[nnw.output_layer])  # a = activations = predictions

    for lr in 1:hp.n_layers
        for f in ff_execstack[lr]
            f(argfilt(dat, nnw, hp, bn, lr, f)...)
        end
    end

end


function feedfwd_predict!(dat::Union{Batch_view, Model_data}, nnw, hp, bn, ff_execstack)
!hp.quiet && println("feedfwd_predict!(dat::Union{Batch_view, Model_data}, nnw, hp)")

    # hidden layers
    # @fastmath @inbounds for hl = 2:nnw.output_layer-1  
    #     affine_function!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
    #     batch_norm_fwd_predict_function!(dat, hl)
    #     unit_function![hl](dat.a[hl], dat.z[hl])
    # end

    # # output layer
    # @inbounds affine!(dat.z[nnw.output_layer], dat.a[nnw.output_layer-1], 
    #                   nnw.theta[nnw.output_layer], nnw.bias[nnw.output_layer])
    # classify_function!(dat.a[nnw.output_layer], dat.z[nnw.output_layer])  # a = activations = predictions

    for lr in 1:hp.n_layers
        for f in ff_execstack[lr]
            if f == getfield(GeneralNN, Symbol("dropout_fwd!"))
                continue
            end
            f(argfilt(dat, nnw, hp, bn, lr, f)...)
        end
    end

end



"""
function backprop!(nnw, dat, hp)
    Argument nnw.delta_th holds the computed gradients for Wgts, delta_b for bias
    Modifies dat.epsilon, nnw.delta_th, nnw.delta_b in place--caller uses nnw.delta_th, nnw.delta_b
    Use for training iterations
    Send it all of the data or a mini-batch
    Intermediate storage of dat.a, dat.z, dat.epsilon, nnw.delta_th, nnw.delta_b reduces memory allocations
"""
function backprop!(nnw, dat, hp)
    !hp.quiet && println("backprop!(nnw, dat, hp)")

    # println("size epsilon of output: ", size(dat.epsilon[nnw.output_layer]))
    # println("size predictions: ", size(dat.a[nnw.output_layer]))
    # println("size targets: ", size(dat.targets))

    # output layer
    @inbounds begin
        # backprop classify
        dat.epsilon[nnw.output_layer][:] = dat.a[nnw.output_layer] .- dat.targets  
            !hp.quiet && println("What is epsilon of output layer? ", mean(dat.epsilon[nnw.output_layer]))
        # backprop affine
        backprop_weights!(nnw.delta_th[nnw.output_layer], nnw.delta_b[nnw.output_layer],  
            dat.epsilon[nnw.output_layer], dat.a[nnw.output_layer-1], hp.mb_size)   
    end

    # loop over hidden layers
    @fastmath @inbounds for hl = (nnw.output_layer - 1):-1:2  
        # backprop activation
        mul!(dat.epsilon[hl], nnw.theta[hl+1]', dat.epsilon[hl+1])
        gradient_function![hl](dat.grad[hl], dat.z[hl])  
            !hp.quiet && println("What is gradient $hl? ", mean(dat.grad[hl]))
        @inbounds dat.epsilon[hl][:] = dat.epsilon[hl] .* dat.grad[hl] 
            !hp.quiet && println("what is epsilon $hl? ", mean(dat.epsilon[hl]))

        batch_norm_back_function!(dat, hl)   # noop if not applicable
        # backprop affine
        backprop_weights_function!(nnw.delta_th[hl], nnw.delta_b[hl], dat.epsilon[hl], dat.a[hl-1], hp.mb_size)
        dropout_back_function![hl](dat, hl)  # noop if not applicable

        !hp.quiet && println("what is delta_th $hl? ", nnw.delta_th[hl])
        !hp.quiet && println("what is delta_b $hl? ", nnw.delta_b[hl])

    end
end


function batch_norm_fwd!(dat, bn, hp, hl; showf=false)
!hp.quiet && println("batch_norm_fwd!(dat, bn, hp, hl)")
showf && begin; println("batch_norm_fwd!"); return; end;

    @inbounds begin
        bn.mu[hl][:] = mean(dat.z[hl], dims=2)          # use in backprop
        bn.stddev[hl][:] = std(dat.z[hl], dims=2)
        dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu[hl]) ./ (bn.stddev[hl] .+ hp.ltl_eps) # normalized: often xhat or zhat  
        dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: often called y 
        bn.mu_run[hl][:] = (  bn.mu_run[hl][1] == 0.0 ? bn.mu[hl] :  
            0.95 .* bn.mu_run[hl] .+ 0.05 .* bn.mu[hl]  )
        bn.std_run[hl][:] = (  bn.std_run[hl][1] == 0.0 ? bn.stddev[hl] :  
            0.95 .* bn.std_run[hl] + 0.05 .* bn.stddev[hl]  )
    end
end


function batch_norm_fwd_predict!(dat, bn, hp, hl)
!hp.quiet && println("batch_norm_fwd_predict!(hp, bn, dat, hl)")

    @inbounds dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu_run[hl]) ./ (bn.std_run[hl] .+ hp.ltl_eps) # normalized: aka xhat or zhat 
    @inbounds dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: often called y 
end


function batch_norm_back!(nnw, dat, bn, hl, hp)
!hp.quiet && println("batch_norm_back!(nnw, dat, bn, hl, hp)")

    mb = hp.mb_size
    @inbounds bn.delta_bet[hl][:] = sum(dat.epsilon[hl], dims=2) ./ mb
    @inbounds bn.delta_gam[hl][:] = sum(dat.epsilon[hl] .* dat.z_norm[hl], dims=2) ./ mb
    @inbounds dat.epsilon[hl][:] = bn.gam[hl] .* dat.epsilon[hl]  # often called delta_z_norm at this stage

    @inbounds dat.epsilon[hl][:] = (                               # often called delta_z, dx, dout, or dy
        (1.0 / mb) .* (1.0 ./ (bn.stddev[hl] .+ hp.ltl_eps))  .* 
            (          
                mb .* dat.epsilon[hl] .- sum(dat.epsilon[hl], dims=2) .-
                dat.z_norm[hl] .* sum(dat.epsilon[hl] .* dat.z_norm[hl], dims=2)
                )
        )
end


function update_parameters!(nnw, hp, bn)  # =Batch_norm_params()
!hp.quiet && println("update_parameters!(nnw, hp, bn)")
    # update Wgts, bias, and batch_norm parameters
    @fastmath @inbounds for hl = 2:nnw.output_layer       
        @inbounds nnw.theta[hl][:] = nnw.theta[hl] .- (hp.alphamod .* nnw.delta_th[hl])
        
        reg_function![hl](nnw, hp, hl)  # regularize function per setup.jl setup_functions!

        # @bp

        if hp.do_batch_norm  # update batch normalization parameters
            @inbounds bn.gam[hl][:] .= bn.gam[hl][:] .- (hp.alphamod .* bn.delta_gam[hl])
            @inbounds bn.bet[hl][:] .= bn.bet[hl][:] .- (hp.alphamod .* bn.delta_bet[hl])
        else  # update bias
            @inbounds nnw.bias[hl][:] .= nnw.bias[hl] .- (hp.alphamod .* nnw.delta_b[hl])
        end

    end  

end


function accuracy(targets, preds)  # this is NOT very general
    if size(targets,1) > 1
        # targetmax = ind2sub(size(targets),vec(findmax(targets,1)[2]))[1]
        # predmax = ind2sub(size(preds),vec(findmax(preds,1)[2]))[1]
        targetmax = getvalidx(targets)     # vec(map(x -> x[1], argmax(targets,dims=1)));
        predmax =   getvalidx(preds)       # vec(map(x -> x[1], argmax(preds,dims=1)));
        fracright = mean(targetmax .== predmax)
    else
        # works because single output unit is classification probability
        # choices = [j > 0.5 ? 1.0 : 0.0 for j in preds]
        choices = zeros(size(preds))
        for i = eachindex(choices)
            choices[i] = preds[i] > 0.5 ? 1.0 : 0.0
        end
        fracright = mean(choices .== targets)
    end
    return fracright
end


function getvalidx(arr, argfunc=argmax)  # could also be argmin
    return vec(map(x -> x[1], argfunc(arr, dims=1)))
end


function r_squared(targets, preds)
    ybar = mean(targets)
    return 1.0 - sum((targets .- preds).^2.) / sum((targets .- ybar).^2.)
end


"""
    Create views for the training data in minibatches
"""
function update_batch_views!(mb::Batch_view, train::Model_data, nnw::Wgts, 
    hp::Hyper_parameters, colrng::UnitRange{Int64})
!hp.quiet && println("update_batch_views!(mb::Batch_view, train::Model_data, nnw::Wgts, 
    hp::Hyper_parameters, colrng::UnitRange{Int64})")

    # colrng refers to the set of training examples included in the minibatch
    n_layers = nnw.output_layer

    # feedforward:   minibatch views update the underlying data
    # TODO put @inbounds back after testing
    @inbounds for i = 1:n_layers
        mb.a[i] = view(train.a[i],:,colrng)   # sel is random order of example indices
        mb.z[i] = view(train.z[i],:,colrng) 
        mb.epsilon[i] = view(train.epsilon[i], :, colrng) 
        mb.grad[i] = view(train.grad[i], :, colrng)  
        # mb.delta_z[i] = view(train.delta_z[i], :, colrng)    
    end
    mb.targets = view(train.targets,:,colrng)  # only at the output layer
    
    if hp.do_batch_norm
        @inbounds for i = 1:n_layers
            # feedforward
            mb.z_norm[i] = view(train.z_norm[i],:, colrng) 
            # backprop
            # mb.delta_z_norm[i] = view(train.delta_z_norm[i], :, colrng)   
        end
    end

    if hp.dropout
        @inbounds for i = 1:n_layers
            # training:  applied to feedforward, but only for training
            mb.dropout_random[i] = view(train.dropout_random[i], :, colrng)  
            mb.dropout_mask_units[i] = view(train.dropout_mask_units[i], :, colrng)  
        end
    end

end

 
function gather_stats!(stats, series, i, dat, nnw, cost_function, hp, bn, ff_execstack)

    if stats["track"][series]
        feedfwd_predict!(dat, nnw, hp, bn, ff_execstack)

        if stats["track"]["cost"]
            stats["cost"][i, stats["col_" * series]] = cost_function(dat.targets,
                dat.a[nnw.output_layer], dat.n, nnw.theta, hp.lambda, hp.reg, nnw.output_layer)
        end
        if stats["track"]["learning"]
            stats["accuracy"][i, stats["col_" * series]] = (  hp.classify == "regression"
                    ? r_squared(dat.targets, dat.a[nnw.output_layer])
                    : accuracy(dat.targets, dat.a[nnw.output_layer])  )
        end
    end
end


# TODO this probably doesn't work any more: not used
function quick_stats(dat, nnw, hp, cost_function=cost_function)

    feedfwd_predict!(dat, nnw, hp, bn)

    cost = cost_function(dat.targets,
            dat.a[nnw.output_layer], dat.n, nnw.theta, hp.lambda, hp.reg, nnw.output_layer)

    correct = (  hp.classify == "regression"
                ? r_squared(dat.targets, dat.a[nnw.output_layer])
                : accuracy(dat.targets, dat.a[nnw.output_layer])  )

    return cost, correct
end
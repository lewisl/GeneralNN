
using StatsBase

# method with no input test data
function training_loop!(hp, train, mb, nnw, bn, statsdat)
    _training_loop!(hp, train, Model_data(), mb, nnw, bn, statsdat)
        # Model_data() passes an empty test data object
end

# method with input of train and test data: test data used for training stats
function training_loop!(hp, train, test, mb, nnw, bn, statsdat)
    _training_loop!(hp, train, test, mb, nnw, bn, statsdat)
end

function _training_loop!(hp, train, test, mb, nnw, bn, statsdat)
!hp.quiet && println("training_loop(hp, train, test mb, nnw, bn, statsdat; dotest=false)")
    
    dotest = isempty(test.inputs) ? false : true

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
                    train_one_step!(mb, nnw, bn, hp, t)

                    # stats for each minibatch--expensive!!!
                    statsdat["period"] == "batch" && begin
                        gather_stats!(statsdat, "train", t, train, nnw, cost_function, hp)  
                        dotest && gather_stats!(statsdat, "test", t, test, nnw, cost_function, hp) 
                    end

                end # mini-batch loop

            else
                t += 1
                train_one_step!(train, nnw, bn, hp, t)
            end

            # stats across all mini-batches of one epoch (e.g.--no stats per minibatch)
            statsdat["period"] == "epoch" && begin
                gather_stats!(statsdat, "train", ep_i, train, nnw, cost_function, hp)  
                dotest && gather_stats!(statsdat, "test", ep_i, test, nnw, cost_function, hp) 
            end

        end # epoch loop
    end # training_time begin block

    return training_time
end # function training_loop


function train_one_step!(dat, nnw, bn, hp, t)

    feedfwd!(dat, nnw, hp)  # for all layers
    backprop!(nnw, dat, hp)  # for all layers   
    optimization_function!(nnw, hp, t)
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
function feedfwd!(dat::Union{Batch_view,Model_data}, nnw, hp)  
!hp.quiet && println("feedfwd!(dat::Union{Batch_view, Model_data}, nnw, hp)")

    # dropout for input layer (if probability < 1.0) or noop
    dropout_fwd_function![1](dat,hp,1)  

    # hidden layers
    @fastmath for hl = 2:nnw.output_layer-1  
        affine_function!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl]) # if do_batch_norm, ignores bias arg
        batch_norm_fwd_function!(dat, hl)  # do it or noop
        unit_function![hl](dat.a[hl], dat.z[hl])
        dropout_fwd_function![hl](dat,hp,hl)  # do it or noop
    end

    # output layer
    @inbounds affine!(dat.z[nnw.output_layer], dat.a[nnw.output_layer-1], 
                      nnw.theta[nnw.output_layer], nnw.bias[nnw.output_layer])
    classify_function!(dat.a[nnw.output_layer], dat.z[nnw.output_layer])  # a = activations = predictions

end


function feedfwd_predict!(dat::Union{Batch_view, Model_data}, nnw, hp)
!hp.quiet && println("feedfwd_predict!(dat::Union{Batch_view, Model_data}, nnw, hp)")

    # hidden layers
    @fastmath for hl = 2:nnw.output_layer-1  
        affine_function!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
        batch_norm_fwd_predict_function!(dat, hl)
        unit_function![hl](dat.a[hl], dat.z[hl])
    end

    # output layer
    @inbounds affine!(dat.z[nnw.output_layer], dat.a[nnw.output_layer-1], 
                      nnw.theta[nnw.output_layer], nnw.bias[nnw.output_layer])
    classify_function!(dat.a[nnw.output_layer], dat.z[nnw.output_layer])  # a = activations = predictions
end



"""
function backprop!(nnw, dat, hp)
    Argument nnw.delta_w holds the computed gradients for Wgts, delta_b for bias
    Modifies dat.epsilon, nnw.delta_w, nnw.delta_b in place--caller uses nnw.delta_w, nnw.delta_b
    Use for training iterations
    Send it all of the data or a mini-batch
    Intermediate storage of dat.a, dat.z, dat.epsilon, nnw.delta_w, nnw.delta_b reduces memory allocations
"""
function backprop!(nnw, dat, hp)
    !hp.quiet && println("backprop!(nnw, dat, hp)")

    # println("size epsilon of output: ", size(dat.epsilon[nnw.output_layer]))
    # println("size predictions: ", size(dat.a[nnw.output_layer]))
    # println("size targets: ", size(dat.targets))

    # for output layer if cross_entropy_cost or mean squared error???    
    dat.epsilon[nnw.output_layer][:] = dat.a[nnw.output_layer] .- dat.targets  
        !hp.quiet && println("What is epsilon of output layer? ", mean(dat.epsilon[nnw.output_layer]))
    backprop_weights!(nnw.delta_w[nnw.output_layer], nnw.delta_b[nnw.output_layer], dat.delta_z[nnw.output_layer], 
        dat.epsilon[nnw.output_layer], dat.a[nnw.output_layer-1], hp.mb_size)      

    # loop over hidden layers
    @fastmath for hl = (nnw.output_layer - 1):-1:2  
        gradient_function![hl](dat.grad[hl], dat.z[hl])  
            !hp.quiet && println("What is gradient $hl? ", mean(dat.grad[hl]))
        mul!(dat.epsilon[hl], nnw.theta[hl+1]', dat.epsilon[hl+1])
        @inbounds dat.epsilon[hl][:] = dat.epsilon[hl] .* dat.grad[hl] 
            !hp.quiet && println("what is epsilon $hl? ", mean(dat.epsilon[hl]))

        # noop if not applicable
        dropout_back_function![hl](dat, hl)
        batch_norm_back_function!(dat, hl)

        backprop_weights_function!(nnw.delta_w[hl], nnw.delta_b[hl], dat.delta_z[hl], 
                                   dat.epsilon[hl], dat.a[hl-1], hp.mb_size)

        !hp.quiet && println("what is delta_w $hl? ", nnw.delta_w[hl])
        !hp.quiet && println("what is delta_b $hl? ", nnw.delta_b[hl])

    end

end


function update_parameters!(nnw, hp, bn=Batch_norm_params())
!hp.quiet && println("update_parameters!(nnw, hp, bn)")
    # update Wgts, bias, and batch_norm parameters
    @fastmath for hl = 2:nnw.output_layer       
        @inbounds nnw.theta[hl][:] = nnw.theta[hl] .- (hp.alpha .* nnw.delta_w[hl])
        
        reg_function![hl](nnw, hp, hl)  # regularize function per setup.jl setup_functions!

        # @bp

        if hp.do_batch_norm  # update batch normalization parameters
            @inbounds bn.gam[hl][:] .= bn.gam[hl][:] .- (hp.alpha .* bn.delta_gam[hl])
            @inbounds bn.bet[hl][:] .= bn.bet[hl][:] .- (hp.alpha .* bn.delta_bet[hl])
        else  # update bias
            @inbounds nnw.bias[hl][:] .= nnw.bias[hl] .- (hp.alpha .* nnw.delta_b[hl])
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
    for i = 1:n_layers
        mb.a[i] = view(train.a[i],:,colrng)   # sel is random order of example indices
        mb.z[i] = view(train.z[i],:,colrng) 
        mb.epsilon[i] = view(train.epsilon[i], :, colrng) 
        mb.grad[i] = view(train.grad[i], :, colrng)   
    end
    mb.targets = view(train.targets,:,colrng)  # only at the output layer
    
    if hp.do_batch_norm
        for i = 1:n_layers
            # feedforward
            mb.z_norm[i] = view(train.z_norm[i],:, colrng) 
            # backprop
            mb.delta_z_norm[i] = view(train.delta_z_norm[i], :, colrng)   
            mb.delta_z[i] = view(train.delta_z[i], :, colrng)   
        end
    end

    if hp.dropout
        for i = 1:n_layers
            # training:  applied to feedforward, but only for training
            mb.dropout_random[i] = view(train.dropout_random[i], :, colrng)  
            mb.dropout_mask_units[i] = view(train.dropout_mask_units[i], :, colrng)  
        end
    end

end


function batch_norm_fwd!(hp, bn, dat, hl)
!hp.quiet && println("batch_norm_fwd!(hp, bn, dat, hl)")

    @inbounds bn.mu[hl][:] = mean(dat.z[hl], dims=2)          # use in backprop
    @inbounds bn.stddev[hl][:] = std(dat.z[hl], dims=2)
    @inbounds dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu[hl]) ./ (bn.stddev[hl] .+ hp.ltl_eps) # normalized: 'aka' xhat or zhat  @inbounds 
    @inbounds dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: 'aka' y  @inbounds 
    @inbounds bn.mu_run[hl][:] = (  bn.mu_run[hl][1] == 0.0 ? bn.mu[hl] :  # @inbounds 
        0.9 .* bn.mu_run[hl] .+ 0.1 .* bn.mu[hl]  )
    @inbounds bn.std_run[hl][:] = (  bn.std_run[hl][1] == 0.0 ? bn.stddev[hl] :  # @inbounds 
        0.9 .* bn.std_run[hl] + 0.1 .* bn.stddev[hl]  )
end


function batch_norm_fwd_predict!(hp, bn, dat, hl)
!hp.quiet && println("batch_norm_fwd_predict!(hp, bn, dat, hl)")

    @inbounds dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu_run[hl]) ./ (bn.std_run[hl] .+ hp.ltl_eps) # normalized: 'aka' xhat or zhat  @inbounds 
    @inbounds dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: 'aka' y  @inbounds 
end


function batch_norm_back!(nnw, dat, bn, hl, hp)
!hp.quiet && println("batch_norm_back!(nnw, dat, bn, hl, hp)")
    mb = hp.mb_size
    @inbounds bn.delta_bet[hl][:] = sum(dat.epsilon[hl], dims=2) ./ mb
    @inbounds bn.delta_gam[hl][:] = sum(dat.epsilon[hl] .* dat.z_norm[hl], dims=2) ./ mb

    @inbounds dat.delta_z_norm[hl][:] = bn.gam[hl] .* dat.epsilon[hl]  

    @inbounds dat.delta_z[hl][:] = (                               
        (1.0 / mb) .* (1.0 ./ bn.stddev[hl]) .* (
            mb .* dat.delta_z_norm[hl] .- sum(dat.delta_z_norm[hl], dims=2) .-
            dat.z_norm[hl] .* sum(dat.delta_z_norm[hl] .* dat.z_norm[hl], dims=2)
            )
        )
end

 
function gather_stats!(statsdat, series, i, dat, nnw, cost_function, hp)

    if statsdat["track"][series]
        feedfwd_predict!(dat, nnw, hp)

        if statsdat["track"]["cost"]
            statsdat["cost"][i, statsdat["col_" * series]] = cost_function(dat.targets,
                dat.a[nnw.output_layer], dat.n, nnw.theta, hp.lambda, hp.reg, nnw.output_layer)
        end
        if statsdat["track"]["learning"]
            statsdat["accuracy"][i, statsdat["col_" * series]] = (  hp.classify == "regression"
                    ? r_squared(dat.targets, dat.a[nnw.output_layer])
                    : accuracy(dat.targets, dat.a[nnw.output_layer])  )
        end
    end

end



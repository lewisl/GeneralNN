# TODO
#   new method for update_parameters with no batch normalization
#   DONE new method for feedfwd
#   DONE new method for backprop
#   DONE new method for update_parameters
#   DONE new method for gather_stats


include("nn_data_structs.jl")

# do minibatch training:  this method accepts both train, mb, bn
function training_loop(hp, datalist, mb, nnp, bn, plotdef)
    if size(datalist, 1) == 1
        train = datalist[1]
        dotest = false
    elseif size(datalist,1) == 2
        train = datalist[1]
        test = datalist[2]
        dotest = true
    else
        error("Datalist contains wrong number of elements.")
    end

training_time = @elapsed begin # start the cpu clock and begin block for training process
    t = 0  # counter:  number of times parameters will have been updated: minibatches * epochs

    for ep_i = 1:hp.epochs  # loop for "epochs" with counter epoch i as ep_i
        !hp.quiet && println("Start epoch $ep_i")
        hp.do_learn_decay && step_learn_decay!(hp, ep_i)

        # reset for at start of each epoch
        done = 0 # how many training examples have been trained on in the epoch
        hp.mb_size = hp.mb_size_in # reset the minibatch size to the input

        for mb_j = 1:hp.n_mb  # loop for mini-batches 
            !hp.quiet && println("   Start minibatch $mb_j")

            # set size of minibatch:  allows for minibatch size that doesn't divide evenly in no. of examples in data
            left = train.n - done  
            hp.mb_size = left > hp.mb_size ? hp.mb_size : left # last minibatch count = left
            done += hp.mb_size
            # set the column range of examples to include in the batch
            first_example = (mb_j - 1) * hp.mb_size + 1  # mini-batch subset for the inputs (layer 1)
            last_example = first_example + hp.mb_size - 1
            colrng = first_example:last_example

            t += 1   # number of executions of minibatch loop
            update_Batch_views!(mb, train, nnp, hp, colrng)  # select data columns for the minibatch                

            feedfwd!(mb, nnp, bn,  hp)  # for all layers

            backprop!(nnp, bn, mb, hp, t)  # for all layers

            optimization_function!(nnp, hp, t)

            update_parameters!(nnp, hp, bn)

        end # mini-batch loop

        # stats across all mini-batches of one epoch (e.g.--no stats per minibatch)
        gather_stats!(plotdef, "train", ep_i, train, nnp, bn, cost_function, hp)  
        dotest && gather_stats!(plotdef, "test", ep_i, test, nnp, cost_function, hp) 


    end # epoch loop
end # begin block for timing

return training_time
end # function training_loop


# this method works for full data training (no minibatches)
# dispatch by removing the mb, bn arguments
function training_loop(hp, datalist, nnp, plotdef)
    if size(datalist, 1) == 1
        train = datalist[1]
        dotest = false
    elseif size(datalist,1) == 2
        train = datalist[1]
        test = datalist[2]
        dotest = true
    else
        error("Datalist contains wrong number of elements.")
    end

training_time = @elapsed begin # start the cpu clock and begin block for training process
    t = 0  # counter:  number of times parameters will have been updated: minibatches * epochs


    for ep_i = 1:hp.epochs  # loop for "epochs" with counter epoch i as ep_i
        !hp.quiet && println("Start epoch $ep_i")
        hp.do_learn_decay && step_learn_decay!(hp, ep_i)

        t += 1 

        feedfwd!(train, nnp, hp)  # for all layers

        backprop!(nnp, train, hp, t)  # for all layers

        optimization_function!(nnp, hp, t)

        update_parameters!(nnp, hp)

        # stats across all mini-batches of one epoch (e.g.--no stats per minibatch)
        gather_stats!(plotdef, "train", ep_i, train, nnp, cost_function, train.n, hp)  
        dotest && gather_stats!(plotdef, "test", ep_i, dat, nnp, cost_function, dat.n, hp) 

    end # epoch loop

end # begin block for timing

return training_time    # don't think we need to return anything because all key functions update in place
end # function training_loop


#########################################################
#  methods for functions inside the training loop
#########################################################


"""
function feedfwd!(dat, nnp, bn, do_batch_norm; istrain)
    modifies a, a_wb, z in place to reduce memory allocations
    send it all of the data or a mini-batch

    feed forward from inputs to output layer predictions
"""
function feedfwd!(dat::Union{Batch_view,Model_data}, nnp, bn,  hp; istrain=true)
    # dropout for input layer (if probability < 1.0)
    if istrain && hp.dropout && (hp.droplim[1] < 1.0)
        dropout!(dat, hp, 1)
    end
    # hidden layers
    @fastmath for hl = 2:nnp.output_layer-1  
        if hp.do_batch_norm 
            affine!(dat.z[hl], dat.a[hl-1], nnp.theta[hl])
            batch_norm_fwd!(hp, bn, dat, hl, istrain)
        else
            affine!(dat.z[hl], dat.a[hl-1], nnp.theta[hl], nnp.bias[hl])
        end

        unit_function!(dat.a[hl], dat.z[hl])

        if istrain && hp.dropout && (hp.droplim[hl] < 1.0)
            dropout!(dat,hp,hl)
        end
    end

    # output layer
    @fastmath @inbounds dat.z[nnp.output_layer][:] = (nnp.theta[nnp.output_layer] * dat.a[nnp.output_layer-1]
        .+ nnp.bias[nnp.output_layer])  # TODO use bias in the output layer with no batch norm? @inbounds 

    classify_function!(dat.a[nnp.output_layer], dat.z[nnp.output_layer])  # a = activations = predictions

end


# dispatches on not have Union for data (excludes Batch_view), not having bn
function feedfwd!(dat::Model_data, nnp, hp; istrain=true)

    # dropout for input layer (if probability < 1.0)
    if istrain && hp.dropout && (hp.droplim[1] < 1.0)
        dropout!(dat, hp, 1)
    end

    # hidden layers
    @fastmath for hl = 2:nnp.output_layer-1  

        affine!(dat.z[hl], dat.a[hl-1], nnp.theta[hl], nnp.bias[hl])
        
        unit_function!(dat.a[hl], dat.z[hl])

        if istrain && hp.dropout && (hp.droplim[hl] < 1.0)
            dropout!(dat,hp,hl)
        end
    end

    # output layer
    @fastmath @inbounds dat.z[nnp.output_layer][:] = (nnp.theta[nnp.output_layer] * dat.a[nnp.output_layer-1]
        .+ nnp.bias[nnp.output_layer])  # TODO use bias in the output layer with no batch norm? @inbounds 

    classify_function!(dat.a[nnp.output_layer], dat.z[nnp.output_layer])  # a = activations = predictions
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

    # for output layer if cross_entropy_cost or mean squared error???
    dat.epsilon[nnp.output_layer][:] = dat.a[nnp.output_layer] .- dat.targets  
    @fastmath nnp.delta_w[nnp.output_layer][:] = dat.epsilon[nnp.output_layer] * dat.a[nnp.output_layer-1]' # 2nd term is effectively the grad for error   
    @fastmath nnp.delta_b[nnp.output_layer][:] = sum(dat.epsilon[nnp.output_layer],dims=2)  

    # loop over hidden layers
    @fastmath for hl = (nnp.output_layer - 1):-1:2  
        gradient_function!(dat.grad[hl], dat.z[hl])
        @inbounds dat.epsilon[hl][:] = nnp.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl] 

        if hp.dropout && (hp.droplim[hl] < 1.0)
            @inbounds dat.epsilon[hl][:] = dat.epsilon[hl] .* dat.dropout_mask_units[hl]
        end

        if hp.do_batch_norm
            batch_norm_back!(nnp, dat, bn, hl, hp)
            @inbounds nnp.delta_w[hl][:] = dat.delta_z[hl] * dat.a[hl-1]'   
        else
            @inbounds nnp.delta_w[hl][:] = dat.epsilon[hl] * dat.a[hl-1]'  
            @inbounds nnp.delta_b[hl][:] = sum(dat.epsilon[hl],dims=2)  #  times a column of 1's = sum(row)
        end

    end

end


# method dispatches on excluding bn argument
function backprop!(nnp, dat, hp, t)

    # for output layer if cross_entropy_cost or mean squared error???
    dat.epsilon[nnp.output_layer][:] = dat.a[nnp.output_layer] .- dat.targets  
    @fastmath nnp.delta_w[nnp.output_layer][:] = dat.epsilon[nnp.output_layer] * dat.a[nnp.output_layer-1]' # 2nd term is effectively the grad for error   
    @fastmath nnp.delta_b[nnp.output_layer][:] = sum(dat.epsilon[nnp.output_layer],dims=2)  

    # loop over hidden layers
    @fastmath for hl = (nnp.output_layer - 1):-1:2  
        @inbounds gradient_function!(dat.grad[hl], dat.z[hl])
        @inbounds dat.epsilon[hl][:] = nnp.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl] 

        if hp.dropout && (hp.droplim[hl] < 1.0)
            @inbounds dat.epsilon[hl][:] = dat.epsilon[hl] .* dat.dropout_mask_units[hl]
        end

        @inbounds nnp.delta_w[hl][:] = dat.epsilon[hl] * dat.a[hl-1]'  
        @inbounds nnp.delta_b[hl][:] = sum(dat.epsilon[hl],dims=2)  #  times a column of 1's = sum(row)

    end

end


function update_parameters!(nnp, hp, bn)
    # update weights, bias, and batch_norm parameters
    @fastmath for hl = 2:nnp.output_layer       

# println(" total change in coeffs ", sum(nnp.delta_w[hl]))

        @inbounds nnp.theta[hl] .= nnp.theta[hl] .- (hp.alphaovermb .* nnp.delta_w[hl])
        if hp.reg == "L2"  # regularization term  opposite sign of gradient?
            @inbounds nnp.theta[hl] .= nnp.theta[hl] .+ (hp.alphaovermb .* (hp.lambda .* nnp.theta[hl]))
        elseif hp.reg == "L1"
            @inbounds nnp.theta[hl] .= nnp.theta[hl] .+ (hp.alphaovermb .* (hp.lambda .* sign.(nnp.theta[hl])))
        end
        
        if hp.do_batch_norm  # update batch normalization parameters
            @inbounds bn.gam[hl][:] -= hp.alphaovermb .* bn.delta_gam[hl]
            @inbounds bn.bet[hl][:] -= hp.alphaovermb .* bn.delta_bet[hl]
        else  # update bias
            @inbounds nnp.bias[hl] .= nnp.bias[hl] .- (hp.alphaovermb .* nnp.delta_b[hl])
        end

    end  

end


# this method dispatches on excluding bn argument
function update_parameters!(nnp, hp)
    
    @fastmath for hl = 2:nnp.output_layer            
        @inbounds nnp.theta[hl] .= nnp.theta[hl] .- (hp.alphaovermb .* nnp.delta_w[hl])

        if hp.reg == "L2"  # regularization term: opposite sign of gradient
            @inbounds nnp.theta[hl] .= nnp.theta[hl] .+ (hp.alphaovermb .* (hp.lambda .* nnp.theta[hl]))
        elseif hp.reg == "L1"
            @inbounds nnp.theta[hl] .= nnp.theta[hl] .+ (hp.alphaovermb .* (hp.lambda  .* sign.(nnp.theta[hl])))
        end
        
        @inbounds nnp.bias[hl] .= nnp.bias[hl] .- (hp.alphaovermb .* nnp.delta_b[hl])
    end  

end


function momentum!(tp, hp, t)
    @fastmath for hl = (tp.output_layer - 1):-1:2  # loop over hidden layers
        @inbounds tp.delta_v_w[hl] .= hp.b1 .* tp.delta_v_w[hl] .+ (1.0 - hp.b1) .* tp.delta_w[hl]  # @inbounds 
        @inbounds tp.delta_w[hl] .= tp.delta_v_w[hl]

        if !hp.do_batch_norm  # then we need to do bias term
            @inbounds tp.delta_v_b[hl] .= hp.b1 .* tp.delta_v_b[hl] .+ (1.0 - hp.b1) .* tp.delta_b[hl]  # @inbounds 
            @inbounds tp.delta_b[hl] .= tp.delta_v_b[hl]
        end
    end
end


function adam!(nnp, hp, t)
    @fastmath for hl = (nnp.output_layer - 1):-1:2  # loop over hidden layers
        @inbounds nnp.delta_v_w[hl] .= hp.b1 .* nnp.delta_v_w[hl] .+ (1.0 - hp.b1) .* nnp.delta_w[hl]  
        @inbounds nnp.delta_s_w[hl] .= hp.b2 .* nnp.delta_s_w[hl] .+ (1.0 - hp.b2) .* nnp.delta_w[hl].^2   
        @inbounds nnp.delta_w[hl] .= (  (nnp.delta_v_w[hl] ./ (1.0 - hp.b1^t)) ./   
                              sqrt.(nnp.delta_s_w[hl] ./ (1.0 - hp.b2^t) .+ hp.ltl_eps)  )

        if !hp.do_batch_norm  # then we need to do bias term
            @inbounds nnp.delta_v_b[hl] .= hp.b1 .* nnp.delta_v_b[hl] .+ (1.0 - hp.b1) .* nnp.delta_b[hl]   
            @inbounds nnp.delta_s_b[hl] .= hp.b2 .* nnp.delta_s_b[hl] .+ (1.0 - hp.b2) .* nnp.delta_b[hl].^2   
            @inbounds nnp.delta_b[hl] .= (  (nnp.delta_v_b[hl] ./ (1.0 - hp.b1^t)) ./   
                              sqrt.(nnp.delta_s_b[hl] ./ (1.0 - hp.b2^t) .+ hp.ltl_eps) )  
        end
    end
end


function no_optimization(tp, hp, t)
end






function accuracy(targets, preds, i)
    if size(targets,1) > 1
        # targetmax = ind2sub(size(targets),vec(findmax(targets,1)[2]))[1]
        # predmax = ind2sub(size(preds),vec(findmax(preds,1)[2]))[1]
        targetmax = vec(map(x -> x[1], argmax(targets,dims=1)));
        predmax = vec(map(x -> x[1], argmax(preds,dims=1)));
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
        choices = [j > 0.5 ? 1.0 : 0.0 for j in preds]
        fracright = mean(choices .== targets)
    end
    return fracright
end


"""
    Create or update views for the training data in minibatches or one big batch
        Arrays: a, z, z_norm, targets  are all fields of struct mb
"""
function update_Batch_views!(mb::Batch_view, train::Model_data, nnp::NN_weights, 
    hp::Hyper_parameters, colrng::UnitRange{Int64})

    # colrng refers to the set of training examples included in the minibatch
    n_layers = nnp.output_layer
    mb_cols = 1:hp.mb_size  # only reason for this is that the last minibatch might be smaller
    
    colselector = hp.shuffle ? mb.sel[colrng] : colrng # note:  always better to do your shuffle before training

    # feedforward:   minibatch views update the underlying data
    @inbounds mb.a = [view(train.a[i],:,colselector) for i = 1:n_layers]  # sel is random order of example indices
    @inbounds mb.targets = view(train.targets,:,colselector)  # only at the output layer
    @inbounds mb.z = [view(train.z[i],:,colselector) for i = 1:n_layers]

    # training / backprop:  don't need this data and only use minibatch size
    @inbounds mb.epsilon = [view(train.epsilon[i], :, mb_cols) for i = 1:n_layers]
    @inbounds mb.grad = [view(train.grad[i], :, mb_cols) for i = 1:n_layers]
    @inbounds mb.delta_z = [view(train.delta_z[i], :, mb_cols) for i = 1:n_layers]

    if hp.do_batch_norm
        # feedforward
        @inbounds mb.z_norm = [view(train.z_norm[i],:, colselector) for i = 1:n_layers]
        # backprop
        @inbounds mb.delta_z_norm = [view(train.delta_z_norm[i], :, mb_cols) for i = 1:n_layers]
    end

    if hp.dropout
        # training:  applied to feedforward, but only for training
        @inbounds mb.dropout_random = [view(train.dropout_random[i], :, mb_cols) for i = 1:n_layers]
        @inbounds mb.dropout_mask_units = [view(train.dropout_mask_units[i], :, mb_cols) for i = 1:n_layers]
    end

end


function batch_norm_fwd!(hp, bn, dat, hl, istrain=true)
    # in_k,mb = size(dat.z[hl])
    if istrain
        @inbounds bn.mu[hl][:] = mean(dat.z[hl], dims=2)          # use in backprop
        @inbounds bn.stddev[hl][:] = std(dat.z[hl], dims=2)
        @inbounds dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu[hl]) ./ (bn.stddev[hl] .+ hp.ltl_eps) # normalized: 'aka' xhat or zhat  @inbounds 
        @inbounds dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: 'aka' y  @inbounds 
        @inbounds bn.mu_run[hl][:] = (  bn.mu_run[hl][1] == 0.0 ? bn.mu[hl] :  # @inbounds 
            0.9 .* bn.mu_run[hl] .+ 0.1 .* bn.mu[hl]  )
        @inbounds bn.std_run[hl][:] = (  bn.std_run[hl][1] == 0.0 ? bn.stddev[hl] :  # @inbounds 
            0.9 .* bn.std_run[hl] + 0.1 .* bn.stddev[hl]  )
    else  # predictions with existing parameters
        @inbounds dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu_run[hl]) ./ (bn.std_run[hl] .+ hp.ltl_eps) # normalized: 'aka' xhat or zhat  @inbounds 
        @inbounds dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: 'aka' y  @inbounds 
    end
end


function batch_norm_back!(nnp, dat, bn, hl, hp)
    mb = hp.mb_size
    @inbounds bn.delta_bet[hl][:] = sum(dat.epsilon[hl], dims=2)
    @inbounds bn.delta_gam[hl][:] = sum(dat.epsilon[hl] .* dat.z_norm[hl], dims=2)

    @inbounds dat.delta_z_norm[hl][:] = bn.gam[hl] .* dat.epsilon[hl]  

    @inbounds dat.delta_z[hl][:] = (                               
        (1.0 / mb) .* (1.0 ./ bn.stddev[hl]) .* (
            mb .* dat.delta_z_norm[hl] .- sum(dat.delta_z_norm[hl], dims=2) .-
            dat.z_norm[hl] .* sum(dat.delta_z_norm[hl] .* dat.z_norm[hl], dims=2)
            )
        )
end

# this method includes bn argument
function gather_stats!(plotdef, train_or_test, i, dat, nnp, bn, cost_function, hp)

    if plotdef["plot_switch"][train_or_test]
        feedfwd!(dat, nnp, bn, hp, istrain=false)

        if plotdef["plot_switch"]["cost"]
            plotdef["cost_history"][i, plotdef[train_or_test]] = cost_function(dat.targets,
                dat.a[nnp.output_layer], dat.n, nnp.theta, hp, nnp.output_layer)
        end
        if plotdef["plot_switch"]["learning"]
            plotdef["fracright_history"][i, plotdef[train_or_test]] = (  hp.classify == "regression"
                    ? r_squared(dat.targets, dat.a[nnp.output_layer])
                    : accuracy(dat.targets, dat.a[nnp.output_layer], i)  )
        end
    end

    # if plotdef["plot_switch"]["test"]
    #     feedfwd!(test, nnp, bn, hp, istrain=false)

    #     if plotdef["plot_switch"]["cost"]
    #         plotdef["cost_history"][i, plotdef["test"]] =cost_function(test.targets,
    #             test.a[nnp.output_layer], test.n, nnp.theta, hp, nnp.output_layer)
    #     end
    #     if plotdef["plot_switch"]["learning"]
    #         plotdef["fracright_history"][i, plotdef["test"]] = (  hp.classify == "regression"
    #                 ? r_squared(test.targets, test.a[nnp.output_layer])
    #                 : accuracy(test.targets, test.a[nnp.output_layer], i)  )
    #     end
    # end
    
end


# this method dispatches on excluding bn argument
function gather_stats!(plotdef, train_or_test, i, dat, nnp, cost_function, hp)

    if plotdef["plot_switch"][train_or_test]
        feedfwd!(dat, nnp, hp, istrain=false)

        if plotdef["plot_switch"]["cost"]
            plotdef["cost_history"][i, plotdef[train_or_test]] = cost_function(dat.targets,
                dat.a[nnp.output_layer], dat.n, nnp.theta, hp, nnp.output_layer)
        end
        if plotdef["plot_switch"]["learning"]
            plotdef["fracright_history"][i, plotdef[train_or_test]] = (  hp.classify == "regression"
                    ? r_squared(dat.targets, dat.a[nnp.output_layer])
                    : accuracy(dat.targets, dat.a[nnp.output_layer], i)  )
        end
    end
    
end
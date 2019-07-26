


include("nn_data_structs.jl")
using StatsBase

# do minibatch training:  this method accepts both train, mb, bn
function training_loop(hp, datalist, mb, nnp, bn, plotdef)
!hp.quiet && println("training_loop(hp, datalist, mb, nnp, bn, plotdef)")
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

            if hp.dobatch
                for mb_j = 1:hp.n_mb  # loop for mini-batches 
                    !hp.quiet && println("   Start minibatch $mb_j")

                    # set size of minibatch:  allows for minibatch size that doesn't divide evenly in no. of examples in data
                    hp.mb_size = mb_j < hp.n_mb ? hp.mb_size_in : hp.last_batch
                    
                    # set the column range of examples to include in the batch
                    first_example = (mb_j - 1) * hp.mb_size_in + 1  # mini-batch subset for the inputs (layer 1)
                    last_example = first_example + hp.mb_size - 1
                    colrng = first_example:last_example

                    t += 1   # number of executions of minibatch loop
                    update_Batch_views!(mb, train, nnp, hp, colrng)  # select data columns for the minibatch   

                    train_one_step!(mb, nnp, bn, hp, t)

                    # stats for each minibatch--expensive!!!
                    hp.plotperbatch && begin
                        gather_stats!(plotdef, "train", t, train, nnp, bn, cost_function, hp)  
                        (dotest && gather_stats!(plotdef, "test", t, test, nnp, bn, cost_function, hp)) 
                    end

                end # mini-batch loop

            else

                train_one_step!(train, nnp, bn, hp, t)

            end

            # stats across all mini-batches of one epoch (e.g.--no stats per minibatch)
            hp.plotperepoch && begin
                gather_stats!(plotdef, "train", ep_i, train, nnp, bn, cost_function, hp)  
                (dotest && gather_stats!(plotdef, "test", ep_i, test, nnp, bn,cost_function, hp)) 
            end

        end # epoch loop
    end # training_time begin block

    return training_time
end # function training_loop


function train_one_step!(dat, nnp, bn, hp, t)

    feedfwd!(dat, nnp, bn,  hp)  # for all layers
    backprop!(nnp, bn, dat, hp)  # for all layers   
    optimization_function!(nnp, hp, t)
    update_parameters!(nnp, hp, bn)

end



#########################################################
#  functions inside the training loop
#########################################################


"""
function feedfwd!(dat, nnp, bn, do_batch_norm; istrain)
    modifies a, a_wb, z in place to reduce memory allocations
    send it all of the data or a mini-batch

    feed forward from inputs to output layer predictions
"""
function feedfwd!(dat::Union{Batch_view,Batch_slice,Model_data}, nnp, bn,  hp; istrain=true)
!hp.quiet && println("feedfwd!(dat::Union{Batch_view,Batch_slice,Model_data}, nnp, bn,  hp; istrain=true)")
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
    @inbounds affine!(dat.z[nnp.output_layer], dat.a[nnp.output_layer-1], 
                      nnp.theta[nnp.output_layer], nnp.bias[nnp.output_layer])

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
function backprop!(nnp, bn, dat, hp)
!hp.quiet && println("backprop!(nnp, bn, dat, hp)")

    # for output layer if cross_entropy_cost or mean squared error???
    dat.epsilon[nnp.output_layer][:] = dat.a[nnp.output_layer] .- dat.targets  
    !hp.quiet && println("What is epsilon of output layer? ", mean(dat.epsilon[nnp.output_layer]))
    @fastmath nnp.delta_w[nnp.output_layer][:] = dat.epsilon[nnp.output_layer] * dat.a[nnp.output_layer-1]' # 2nd term is effectively the grad for error   
    @fastmath nnp.delta_b[nnp.output_layer][:] = sum(dat.epsilon[nnp.output_layer],dims=2)  

    # loop over hidden layers
    @fastmath for hl = (nnp.output_layer - 1):-1:2  
        gradient_function!(dat.grad[hl], dat.z[hl])
        !hp.quiet && println("What is gradient $hl? ", mean(dat.grad[hl]))
        @inbounds dat.epsilon[hl][:] = nnp.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl] 
        !hp.quiet && println("what is epsilon $hl? ", mean(dat.epsilon[hl]))

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

        !hp.quiet && println("what is delta_w $hl? ", mean(nnp.delta_w[hl]))
        !hp.quiet && println("what is delta_b $hl? ", mean(nnp.delta_w[hl]))

    end

end


function update_parameters!(nnp, hp, bn)
!hp.quiet && println("update_parameters!(nnp, hp, bn)")
    # update weights, bias, and batch_norm parameters
    @fastmath for hl = 2:nnp.output_layer       
        @inbounds nnp.theta[hl] .= nnp.theta[hl] .- (hp.alphaovermb .* nnp.delta_w[hl])

        if hp.reg == "L2"  # regularization term  opposite sign of gradient?
            @inbounds nnp.theta[hl] .= nnp.theta[hl] .+ (hp.alphaovermb .* (hp.lambda .* nnp.theta[hl]))
        elseif hp.reg == "L1"
            @inbounds nnp.theta[hl] .= nnp.theta[hl] .+ (hp.alphaovermb .* (hp.lambda .* sign.(nnp.theta[hl])))
        elseif hp.reg == "Maxnorm"
            maxnorm_reg!(nnp.theta[hl], hp.maxnorm_lim[hl])
        end
        
        if hp.do_batch_norm  # update batch normalization parameters
            @inbounds bn.gam[hl][:] -= hp.alphaovermb .* bn.delta_gam[hl]
            @inbounds bn.bet[hl][:] -= hp.alphaovermb .* bn.delta_bet[hl]
        else  # update bias
            @inbounds nnp.bias[hl] .= nnp.bias[hl] .- (hp.alphaovermb .* nnp.delta_b[hl])
        end

    end  

end


function accuracy(targets, preds)
    if size(targets,1) > 1
        # targetmax = ind2sub(size(targets),vec(findmax(targets,1)[2]))[1]
        # predmax = ind2sub(size(preds),vec(findmax(preds,1)[2]))[1]
        targetmax = vec(map(x -> x[1], argmax(targets,dims=1)));
        predmax = vec(map(x -> x[1], argmax(preds,dims=1)));
        fracright = mean(targetmax .== predmax)
    else
        # works because single output unit is sigmoid
        choices = [j > 0.5 ? 1.0 : 0.0 for j in preds]
        fracright = mean(choices .== targets)
    end
    return fracright
end


function r_squared(targets, preds)
    ybar = mean(targets)
    return 1.0 - sum((targets .- preds).^2.) / sum((targets .- ybar).^2.)
end



"""
    Create views for the training data in minibatches
"""
function update_Batch_views!(mb::Batch_view, train::Model_data, nnp::NN_weights, 
    hp::Hyper_parameters, colrng::UnitRange{Int64})
!hp.quiet && println("update_Batch_views!(mb::Batch_view, train::Model_data, nnp::NN_weights, 
    hp::Hyper_parameters, colrng::UnitRange{Int64})")

    # colrng refers to the set of training examples included in the minibatch
    n_layers = nnp.output_layer
    mb_cols = 1:hp.mb_size  # only reason for this is that the last minibatch might be smaller
    
    # TODO this is really bad for sparsearrays and generally results in slow indexing
    colselector = hp.shuffle ? mb.sel[colrng] : colrng # note:  always better to do your shuffle before training

    # feedforward:   minibatch views update the underlying data
    @inbounds mb.a = [view(train.a[i],:,colselector) for i = 1:n_layers]  # sel is random order of example indices
    @inbounds mb.targets = view(train.targets,:,colselector)  # only at the output layer
    @inbounds mb.z = [view(train.z[i],:,colselector) for i = 1:n_layers]

    # training / backprop:  don't need this data and only use minibatch size
    # TEST
    @inbounds mb.epsilon = [view(train.epsilon[i], :, mb_cols) for i = 1:n_layers]
    @inbounds mb.grad = [view(train.grad[i], :, mb_cols) for i = 1:n_layers]
    

    if hp.do_batch_norm
        # feedforward
        @inbounds mb.z_norm = [view(train.z_norm[i],:, colselector) for i = 1:n_layers]
        # backprop
        @inbounds mb.delta_z_norm = [view(train.delta_z_norm[i], :, mb_cols) for i = 1:n_layers]
        @inbounds mb.delta_z = [view(train.delta_z[i], :, mb_cols) for i = 1:n_layers]
    end

    if hp.dropout
        # training:  applied to feedforward, but only for training
        @inbounds mb.dropout_random = [view(train.dropout_random[i], :, mb_cols) for i = 1:n_layers]
        @inbounds mb.dropout_mask_units = [view(train.dropout_mask_units[i], :, mb_cols) for i = 1:n_layers]
    end

end


function batch_norm_fwd!(hp, bn, dat, hl, istrain=true)
!hp.quiet && println("batch_norm_fwd!(hp, bn, dat, hl, istrain=true)")
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
!hp.quiet && println("batch_norm_back!(nnp, dat, bn, hl, hp)")
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
            plotdef["accuracy"][i, plotdef[train_or_test]] = (  hp.classify == "regression"
                    ? r_squared(dat.targets, dat.a[nnp.output_layer])
                    : accuracy(dat.targets, dat.a[nnp.output_layer])  )
        end
    end

end


##############################################################################
#  this is a slice approach for performance comparison
##############################################################################

# # This method dispatches on minibatches that are slices
# function update_Batch_views!(mb::Batch_slice, train::Model_data, nnp::NN_weights, 
#     hp::Hyper_parameters, colrng::UnitRange{Int64})
# !hp.quiet && println("update_Batch_views!(mb::Batch_slice, train::Model_data, nnp::NN_weights, 
#     hp::Hyper_parameters, colrng::UnitRange{Int64})")

#     # colrng refers to the set of training examples included in the minibatch
#     n_layers = nnp.output_layer
#     mb_cols = 1:hp.mb_size  # only reason for this is that the last minibatch might be smaller
    
#     # TODO this is really bad for sparsearrays and generally results in slow indexing
#     colselector = hp.shuffle ? mb.sel[colrng] : colrng # note:  always better to do your shuffle before training

#     # feedforward:   minibatch slices update the underlying data
#     @inbounds for i = 1:n_layers
#         mb.a[i][:] = train.a[i][:,colselector]
#         mb.targets[:] = train.targets[:,colselector]  
#         mb.z[i][:] = train.z[i][:,colselector]

#         # training / backprop:  don't need this data and only use minibatch size
#         mb.epsilon[i][:] = train.epsilon[i][:, mb_cols]
#         mb.grad[i][:] = train.grad[i][:, mb_cols]
#         mb.delta_z[i][:] = train.delta_z[i][:, mb_cols]

#         if hp.do_batch_norm
#             # feedforward
#             mb.z_norm[i][:] = train.z_norm[i][:, colselector]
#             # backprop
#             mb.delta_z_norm[i][:] = train.delta_z_norm[i][:, mb_cols]
#         end

#         if hp.dropout
#             # training:  applied to feedforward, but only for training
#             mb.dropout_random[i][:] = train.dropout_random[i][:, mb_cols]
#             mb.dropout_mask_units[i][:] = train.dropout_mask_units[i][:, mb_cols]
#         end
#     end
# end


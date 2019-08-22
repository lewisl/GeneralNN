
using StatsBase

function training_loop!(hp, datalist, mb, nnw, bn, plotdef)
!hp.quiet && println("training_loop(hp, datalist, mb, nnw, bn, plotdef)")
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
        mbszin = hp.mb_size_in

        for ep_i = 1:hp.epochs  # loop for "epochs" with counter epoch i as ep_i
            !hp.quiet && println("Start epoch $ep_i")
            hp.do_learn_decay && step_learn_decay!(hp, ep_i)

            if hp.dobatch  # minibatch training

                for colrng in MBrng(train.n, mbszin)  # set setup.jl for definition of iterator
                    hp.mb_size = size(colrng, 1)   
                    !hp.quiet && println("   Start minibatch for ", colrng)           
                    
                    update_batch_views!(mb, train, nnw, hp, colrng)  # select data columns for the minibatch   

                    t += 1   # number of executions of minibatch loop
                    train_one_step!(mb, nnw, bn, hp, t)

                    # stats for each minibatch--expensive!!!
                    hp.plotperbatch && begin
                        gather_stats!(plotdef, "train", t, train, nnw, bn, cost_function, hp)  
                        dotest && gather_stats!(plotdef, "test", t, test, nnw, bn, cost_function, hp) 
                    end

                end # mini-batch loop

            else

                t += 1
                train_one_step!(train, nnw, bn, hp, t)

            end

            # stats across all mini-batches of one epoch (e.g.--no stats per minibatch)
            hp.plotperepoch && begin
                gather_stats!(plotdef, "train", ep_i, train, nnw, bn, cost_function, hp)  
                dotest && gather_stats!(plotdef, "test", ep_i, test, nnw, bn,cost_function, hp) 
            end

        end # epoch loop
    end # training_time begin block

    return training_time
end # function training_loop


function train_one_step!(dat, nnw, bn, hp, t)

    feedfwd!(dat, nnw, bn,  hp)  # for all layers
    backprop!(nnw, bn, dat, hp)  # for all layers   
    optimization_function!(nnw, hp, t)
    update_parameters!(nnw, hp, bn)

end



#########################################################
#  functions inside the training loop
#########################################################


"""
function feedfwd!(dat, nnw, bn, do_batch_norm; istrain)
    modifies a, a_wb, z in place to reduce memory allocations
    send it all of the data or a mini-batch

    feed forward from inputs to output layer predictions
"""
function feedfwd!(dat::Union{Batch_view,Batch_slice,Model_data}, nnw, bn,  hp; istrain=true)
!hp.quiet && println("feedfwd!(dat::Union{Batch_view,Batch_slice,Model_data}, nnw, bn,  hp; istrain=true)")

    # dropout for input layer (if probability < 1.0)
    if istrain && hp.dropout && (hp.droplim[1] < 1.0)
        dropout!(dat, hp, 1)
    end

    # hidden layers
    @fastmath for hl = 2:nnw.output_layer-1  
        if hp.do_batch_norm 
            affine!(dat.z[hl], dat.a[hl-1], nnw.theta[hl])
            batch_norm_fwd!(hp, bn, dat, hl, istrain)
        else
            affine!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
        end

        unit_function!(dat.a[hl], dat.z[hl])

        if istrain && hp.dropout && (hp.droplim[hl] < 1.0)
            dropout!(dat,hp,hl)
        end
    end

    # output layer
    @inbounds affine!(dat.z[nnw.output_layer], dat.a[nnw.output_layer-1], 
                      nnw.theta[nnw.output_layer], nnw.bias[nnw.output_layer])

    classify_function!(dat.a[nnw.output_layer], dat.z[nnw.output_layer])  # a = activations = predictions

end


"""
function backprop!(nnw, dat, do_batch_norm)
    Argument nnw.delta_w holds the computed gradients for weights, delta_b for bias
    Modifies dat.epsilon, nnw.delta_w, nnw.delta_b in place--caller uses nnw.delta_w, nnw.delta_b
    Use for training iterations
    Send it all of the data or a mini-batch
    Intermediate storage of dat.a, dat.z, dat.epsilon, nnw.delta_w, nnw.delta_b reduces memory allocations
"""
function backprop!(nnw, bn, dat, hp)
!hp.quiet && println("backprop!(nnw, bn, dat, hp)")

    # for output layer if cross_entropy_cost or mean squared error???
    dat.epsilon[nnw.output_layer][:] = dat.a[nnw.output_layer] .- dat.targets  
    !hp.quiet && println("What is epsilon of output layer? ", mean(dat.epsilon[nnw.output_layer]))
    @fastmath nnw.delta_w[nnw.output_layer][:] = dat.epsilon[nnw.output_layer] * dat.a[nnw.output_layer-1]' # 2nd term is effectively the grad for error   
    @fastmath nnw.delta_b[nnw.output_layer][:] = sum(dat.epsilon[nnw.output_layer],dims=2)  

    # loop over hidden layers
    @fastmath for hl = (nnw.output_layer - 1):-1:2  
        gradient_function!(dat.grad[hl], dat.z[hl])
        !hp.quiet && println("What is gradient $hl? ", mean(dat.grad[hl]))
        @inbounds dat.epsilon[hl][:] = nnw.theta[hl+1]' * dat.epsilon[hl+1] .* dat.grad[hl] 
        !hp.quiet && println("what is epsilon $hl? ", mean(dat.epsilon[hl]))

        if hp.dropout && (hp.droplim[hl] < 1.0)
            @inbounds dat.epsilon[hl][:] = dat.epsilon[hl] .* dat.dropout_mask_units[hl]
        end

        if hp.do_batch_norm
            batch_norm_back!(nnw, dat, bn, hl, hp)
            @inbounds nnw.delta_w[hl][:] = dat.delta_z[hl] * dat.a[hl-1]'  

        else
            @inbounds nnw.delta_w[hl][:] = dat.epsilon[hl] * dat.a[hl-1]'  
            @inbounds nnw.delta_b[hl][:] = sum(dat.epsilon[hl],dims=2)  #  times a column of 1's = sum(row)
        end

        !hp.quiet && println("what is delta_w $hl? ", mean(nnw.delta_w[hl]))
        !hp.quiet && println("what is delta_b $hl? ", mean(nnw.delta_w[hl]))

    end

end


function update_parameters!(nnw, hp, bn)
!hp.quiet && println("update_parameters!(nnw, hp, bn)")
    # update weights, bias, and batch_norm parameters
    @fastmath for hl = 2:nnw.output_layer       
        @inbounds nnw.theta[hl] .= nnw.theta[hl] .- (hp.alphaovermb .* nnw.delta_w[hl])
        
        reg_function!(nnw, hp, hl)  # regularize function per setup.jl setup_functions!

        if hp.do_batch_norm  # update batch normalization parameters
            @inbounds bn.gam[hl][:] .= bn.gam[hl][:] .- (hp.alphaovermb .* bn.delta_gam[hl])
            @inbounds bn.bet[hl][:] .= bn.bet[hl][:] .- (hp.alphaovermb .* bn.delta_bet[hl])
        else  # update bias
            @inbounds nnw.bias[hl] .= nnw.bias[hl] .- (hp.alphaovermb .* nnw.delta_b[hl])
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
function update_batch_views!(mb::Batch_view, train::Model_data, nnw::NN_weights, 
    hp::Hyper_parameters, colrng::UnitRange{Int64})
!hp.quiet && println("update_batch_views!(mb::Batch_view, train::Model_data, nnw::NN_weights, 
    hp::Hyper_parameters, colrng::UnitRange{Int64})")

    # colrng refers to the set of training examples included in the minibatch
    n_layers = nnw.output_layer
    mb_cols = 1:hp.mb_size  # only reason for this is that the last minibatch might be smaller

    # feedforward:   minibatch views update the underlying data
    # TODO put @inbounds back after testing
    for i = 1:n_layers
        mb.a[i] = view(train.a[i],:,colrng)   # sel is random order of example indices
        mb.z[i] = view(train.z[i],:,colrng) 
        mb.epsilon[i] = view(train.epsilon[i], :, mb_cols) # NOTE: use mb_cols!
        mb.grad[i] = view(train.grad[i], :, mb_cols) 
    end
    mb.targets = view(train.targets,:,colrng)  # only at the output layer

    # training / backprop:  don't need this data and only use minibatch size
    # TEST
    
    if hp.do_batch_norm
        for i = 1:n_layers
            # feedforward
            mb.z_norm[i] = view(train.z_norm[i],:, colrng) 
            # backprop
            mb.delta_z_norm[i] = view(train.delta_z_norm[i], :, mb_cols) 
            mb.delta_z[i] = view(train.delta_z[i], :, mb_cols) 
        end
    end

    if hp.dropout
        for i = 1:n_layers
            # training:  applied to feedforward, but only for training
            mb.dropout_random[i] = view(train.dropout_random[i], :, mb_cols)
            mb.dropout_mask_units[i] = view(train.dropout_mask_units[i], :, mb_cols)
        end
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


function batch_norm_back!(nnw, dat, bn, hl, hp)
!hp.quiet && println("batch_norm_back!(nnw, dat, bn, hl, hp)")
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
function gather_stats!(plotdef, train_or_test, i, dat, nnw, bn, cost_function, hp)

    if plotdef["plot_switch"][train_or_test]
        feedfwd!(dat, nnw, bn, hp, istrain=false)

        if plotdef["plot_switch"]["cost"]
            plotdef["cost_history"][i, plotdef[train_or_test]] = cost_function(dat.targets,
                dat.a[nnw.output_layer], dat.n, nnw.theta, hp, nnw.output_layer)
        end
        if plotdef["plot_switch"]["learning"]
            plotdef["accuracy"][i, plotdef[train_or_test]] = (  hp.classify == "regression"
                    ? r_squared(dat.targets, dat.a[nnw.output_layer])
                    : accuracy(dat.targets, dat.a[nnw.output_layer])  )
        end
    end

end


##############################################################################
#  this is a slice approach for performance comparison
##############################################################################

# # This method dispatches on minibatches that are slices
# function update_batch_views!(mb::Batch_slice, train::Model_data, nnw::NN_weights, 
#     hp::Hyper_parameters, colrng::UnitRange{Int64})
# !hp.quiet && println("update_batch_views!(mb::Batch_slice, train::Model_data, nnw::NN_weights, 
#     hp::Hyper_parameters, colrng::UnitRange{Int64})")

#     # colrng refers to the set of training examples included in the minibatch
#     n_layers = nnw.output_layer
#     mb_cols = 1:hp.mb_size  # only reason for this is that the last minibatch might be smaller

#     # feedforward:   minibatch slices update the underlying data
#     @inbounds for i = 1:n_layers
#         mb.a[i][:] = train.a[i][:,colrng]
#         mb.targets[:] = train.targets[:,colrng]  
#         mb.z[i][:] = train.z[i][:,colrng]

#         # training / backprop:  don't need this data and only use minibatch size
#         mb.epsilon[i][:] = train.epsilon[i][:, mb_cols]
#         mb.grad[i][:] = train.grad[i][:, mb_cols]
#         mb.delta_z[i][:] = train.delta_z[i][:, mb_cols]

#         if hp.do_batch_norm
#             # feedforward
#             mb.z_norm[i][:] = train.z_norm[i][:, colrng]
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


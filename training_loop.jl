
using StatsBase # basic statistical functions


"""
    training_loop!(hp, train, mb, nnw, bn, stats, model)
    training_loop!(hp, train, test, mb, nnw, bn, stats, model)

    Inputs:
    hp:       Hyper_parameters object
    train:    Model_data object
    mb:       Batch_view object (for minibatches--provide one even 
                   if it is empty and unused)
    nnw:      Wgts (trained parameters) object
    bn:       Batch_norm_params object
    stats:    Dict created by setup_stats function to hold training statistics
    model:    Object containing definition of model

Performs machine learning training using gradient descent. Enables minibatch learning and stochastic gradient
descent with a batch size of 1. The full loop includes feed forward, back propagation, optimization of 
parameter updates and updating the trained parameters. The first method does not include test data. The second method
includes the test Model_data object to track training statistics on how cost and accuracy change for the 
test or validation data set.
"""
function training_loop!(hp, train, test, mb, nnw, bn, stats, model, train_method)
!hp.quiet && println("training_loop(hp, train, test mb, nnw, bn, stats; dotest=false)")

    print_model(model); println()

    training_time = @elapsed begin # start the cpu clock and begin block for training process
        # startup
        hp.alphamod = hp.alpha # set alphamod which is actually used as the learning rate

        train_method(hp, train, test, mb, nnw, bn, stats, model)

    end # training_time begin block

    return training_time
end # function training_loop


function minibatch_training(hp, train, test, mb, nnw, bn, stats, model)

    dotest = isempty(test.inputs) ? false : true

    t = 0  # counter:  number of times parameters will have been updated: minibatches * epochs

    for ep_i = 1:hp.epochs  # loop for "epochs" with counter epoch i as ep_i
        !hp.quiet && println("Start epoch $ep_i")
        hp.do_learn_decay && step_learn_decay!(hp, ep_i)

        hp.reshuffle && (ep_i % 2 == 0 && shuffle_data!(train.inputs, train.targets))

        for colrng in MBrng(train.n, hp.mb_size_in)  # set setup_model.jl for definition of iterator MBrng
            hp.mb_size = mbsize(colrng)

            !hp.quiet && println("   Start minibatch for ", colrng)           
            
            update_batch_views!(mb, train, nnw, hp, colrng)  # select data columns for the minibatch   

            t += 1   # number of executions of minibatch loop
            train_one_step!(mb, nnw, bn, hp, t, model)

            # stats for each minibatch--expensive!!!
            stats["period"] == "batch" && begin
                gather_stats!(stats, "train", t, train, nnw, hp, bn, model)  
                dotest && gather_stats!(stats, "test", t, test, nnw, hp, bn, model) 
            end

        end # mini-batch loop

        # stats across all mini-batches of one epoch (e.g.--no stats per minibatch)
        stats["period"] == "epoch" && begin
            gather_stats!(stats, "train", ep_i, train, nnw, hp, bn, model)  
            dotest && gather_stats!(stats, "test", ep_i, test, nnw, hp, bn, model) 
        end

    end # epoch loop
end


function fullbatch_training(hp, train, test, mb, nnw, bn, stats, model)

    dotest = isempty(test.inputs) ? false : true

    t = 0  # counter:  number of times parameters will have been updated: minibatches * epochs

    for ep_i = 1:hp.epochs  # loop for "epochs" with counter epoch i as ep_i
        !hp.quiet && println("Start epoch $ep_i")
        hp.do_learn_decay && step_learn_decay!(hp, ep_i)

        hp.reshuffle && (ep_i % 2 == 0 && shuffle_data!(train.inputs, train.targets))

        t += 1

        train_one_step!(train, nnw, bn, hp, t, model)

        # stats across all mini-batches of one epoch (e.g.--no stats per minibatch)
        stats["period"] == "epoch" && begin
            gather_stats!(stats, "train", ep_i, train, nnw, hp, bn, model)  
            dotest && gather_stats!(stats, "test", ep_i, test, nnw, hp, bn, model) 
        end

    end # epoch loop
end


# function train_one_step!(dat, nnw, bn, hp, t)
function train_one_step!(dat, nnw, bn, hp, t, model)

    feedfwd!(dat, nnw, hp, bn, model.ff_execstack)  
    backprop!(nnw, dat, hp, bn, model.back_execstack)     
    update_parameters!(nnw, hp, bn, t, model.update_execstack)

end

####################################################
# batch training helper functions
####################################################

    mbsize(colrng) = float(size(colrng, 1))

    # iterator for minibatches of training examples
    struct MBrng  # values are set once to define the iterator stop point=cnt, and increment=incr
        cnt::Int
        incr::Int
    end

    function mbiter(mb::MBrng, start)  # new method for Base.iterate
        nxtstart = start + mb.incr
        stop = nxtstart - 1 < mb.cnt ? nxtstart - 1 : mb.cnt
        ret = start < mb.cnt ? (start:stop, nxtstart) : nothing # return tuple of range and next state, or nothing--to stop iteration
        return ret
    end

    function mblength(mb::MBrng)  # new method for Base.length
        return ceil(Int,mb.cnt / mb.incr)
    end

    # add iterate methods: must supply type for the new methods--method dispatch selects the method for this type of iterator
        # the function  names don't matter--we provide an alternate for the standard methods, but the functions
        # need to do the right things
    Base.iterate(mb::MBrng, start=1) = mbiter(mb::MBrng, start)   # canonical to use "state" instead of "start"
    Base.length(mb::MBrng) = mblength(mb)


#########################################################
#  functions inside the training loop
#########################################################


"""
function feedfwd!(dat, nnw, do_batch_norm)
    modifies a, a_wb, z in place to reduce memory allocations
    send it all of the data or a mini-batch

    feed forward from inputs to output layer predictions
"""
function feedfwd!(dat::Union{Batch_view,Model_data}, nnw, hp, bn, ff_execstack; dotrain=true)  
!hp.quiet && println("feedfwd!(dat::Union{Batch_view, Model_data}, nnw, hp)")

    @simd for lr in 1:hp.n_layers
        @simd for f in ff_execstack[lr]
            f(dat, nnw, hp, bn, lr, dotrain) 
            # use the same args for everything: f calls its method in layer_functions.jl with needed inputs
        end
    end

end


"""
function backprop!(nnw, dat, hp, bn, back_execstack)
    Argument nnw.delta_th holds the computed gradients for Wgts, delta_b for bias
    Modifies dat.epsilon, nnw.delta_th, nnw.delta_b in place--caller uses nnw.delta_th, nnw.delta_b
    Use for training iterations
    Send it all of the data or a mini-batch
    Intermediate storage of dat.a, dat.z, dat.epsilon, nnw.delta_th, nnw.delta_b reduces memory allocations
"""
function backprop!(nnw::Wgts, dat::Union{Batch_view,Model_data}, hp, bn, back_execstack)
    !hp.quiet && println("backprop!(nnw, dat, hp)")

    @simd for lr in hp.n_layers:-1:1
        @simd for f in back_execstack[lr]
            f(dat, nnw, hp, bn, lr) 
            # use the same args for everything: f calls its method in layer_functions.jl with needed inputs
        end
    end

    !hp.quiet && println("what is delta_th $hl? ", nnw.delta_th[hl])
    !hp.quiet && println("what is delta_b $hl? ", nnw.delta_b[hl])

end


function update_parameters!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, t::Int, update_execstack)  # =Batch_norm_params()
!hp.quiet && println("update_parameters!(nnw, hp, bn)")

    @simd for lr in hp.n_layers:-1:1
        @simd for f in update_execstack[lr]
            f(nnw, hp, bn, lr, t) 
            # use the same args for everything: f calls its method in layer_functions.jl with needed inputs
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

end

 
function gather_stats!(stats, series, i, dat, nnw, hp, bn, model)

    if stats["track"][series]
        feedfwd!(dat, nnw, hp, bn, model.ff_execstack, dotrain=false)

        if stats["track"]["cost"]
            stats["cost"][i, stats["col_" * series]] = model.cost_function(dat.targets,
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
function quick_stats(dat, nnw, hp, model)

    feedfwd(dat, nnw, hp, bn, model.ff_execstack, dotrain=false)

    cost = model.cost_function(dat.targets,
            dat.a[nnw.output_layer], dat.n, nnw.theta, hp.lambda, hp.reg, nnw.output_layer)

    correct = (  hp.classify == "regression"
                ? r_squared(dat.targets, dat.a[nnw.output_layer])
                : accuracy(dat.targets, dat.a[nnw.output_layer])  )

    return cost, correct
end
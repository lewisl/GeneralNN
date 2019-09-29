using Plots
using JLD2
using Printf
using LinearAlgebra




function normalize_inputs!(inputs, norm_mode="none")
    if lowercase(norm_mode) == "standard"
        # normalize training data
        x_mu = mean(inputs, dims=2)
        x_std = std(inputs, dims=2)
        inputs[:] = (inputs .- x_mu) ./ (x_std .+ 1e-08)
        norm_factors = (x_mu, x_std) # tuple of Array{Float64,2}
    elseif lowercase(norm_mode) == "minmax"
        # normalize training data
        x_max = maximum(inputs, dims=2)
        x_min = minimum(inputs, dims=2)
        inputs[:] = (inputs .- x_min) ./ (x_max .- x_min .+ 1e-08)
        norm_factors = (x_min, x_max) # tuple of Array{Float64,2}
    else  # handles case of "", "none" or really any crazy string
        norm_factors = ([0.0], [1.0])
    end

    # to translate to unnormalized regression coefficients: m = mhat / stdx, b = bhat - (m*xmu)
    # precalculate a and b constants, and 
    # then just apply newvalue = a * value + b. a = (max'-min')/(max-min) and b = max - a * max 
    # (x - x.min()) / (x.max() - x.min())       # values from 0 to 1
    # 2*(x - x.min()) / (x.max() - x.min()) - 1 # values from -1 to 1

    return norm_factors
end

# apply previously used training normalization to a validation or test data set
function normalize_inputs!(inputs, norm_factors, norm_mode)
    if norm_mode == "standard"
        x_mu = norm_factors[1]
        x_std = norm_factors[2]
        inputs[:] = (inputs .- x_mu) ./ (x_std .+ 1e-08)
    elseif norm_mode == "minmax"
        x_min = norm_factors[1]
        x_max = norm_factors[2]
        inputs[:] = (inputs .- x_min) ./ (x_max .- x_min .+ 1e-08)
    else
        error("Input norm_mode = $norm_mode must be standard or minmax")
    end
end


####################################################################
#  functions to pre-allocate data updated during training loop
####################################################################

# use for test and training data
function preallocate_data!(dat, nnw, n, hp; istrain=true)

    # feedforward
    dat.a = [dat.inputs]  # allocates only tiny memory--it's a reference
    dat.z = [dat.inputs] # not used for input layer  TODO--this permeates the code but not needed
    if hp.sparse
        for i = 2:nnw.output_layer  
            push!(dat.z, spzeros(nnw.ks[i], n, 0.1))
            push!(dat.a, spzeros(nnw.ks[i], n, 0.1))  #  and up...  ...output layer set after loop
        end
    else
        for i = 2:nnw.output_layer  
            push!(dat.z, zeros(nnw.ks[i], n))
            push!(dat.a, zeros(nnw.ks[i], n))  #  and up...  ...output layer set after loop
        end
    end

    # training / backprop  -- pre-allocate only minibatch size (except last one, which could be smaller)
    # this doesn't work for test set when not using minibatches (minibatch size on training then > entire test set)
    if istrain   # e.g., only for training->no backprop data structures needed for test data
        if hp.dobatch   # TODO  fix this HACK
            dat.epsilon = [i[:,1:hp.mb_size_in] for i in dat.a]
            dat.grad = [i[:,1:hp.mb_size_in] for i in dat.a]
            dat.delta_z = [i[:,1:hp.mb_size_in] for i in dat.a]
        else  # this should pick up sparsity
            dat.epsilon = [i for i in dat.a]
            dat.grad = [i for i in dat.a]
            dat.delta_z = [i for i in dat.a]
        end
    end

    if hp.do_batch_norm  # required for full pass performance stats  TODO: really? or only for batch_norm
        # feedforward
        dat.z_norm = deepcopy(dat.z)
        # backprop
        dat.delta_z_norm = deepcopy(dat.z)
        # preallocate_batchnorm!(bn, mb, nnw.ks)
    end

    # backprop / training
    if hp.dropout
        dat.dropout_random = [i[:,1:hp.mb_size_in] for i in dat.a]
        dat.dropout_mask_units = [BitArray(ones(size(i,1),hp.mb_size_in)) for i in dat.a]
    end

end



# method for batch views--currently the only method used
function preallocate_minibatch!(mb::Batch_view, nnw, hp)
    # feedforward:   minibatch views update the underlying data
    # TODO put @inbounds back after testing
    n_layers = nnw.output_layer

    # we don't need all of these depending on minibatches and batchnorm, but it's very little memory
    mb.a = Array{SubArray{}}(undef, n_layers)
    mb.targets = view([0.0],1:1)
    mb.z = Array{SubArray{}}(undef, n_layers)
    mb.z_norm = Array{SubArray{}}(undef, n_layers)
    mb.delta_z_norm = Array{SubArray{}}(undef, n_layers)
    mb.delta_z = Array{SubArray{}}(undef, n_layers)
    mb.grad = Array{SubArray{}}(undef, n_layers)
    mb.epsilon = Array{SubArray{}}(undef, n_layers)
    mb.dropout_random = Array{SubArray{}}(undef, n_layers)
    mb.dropout_mask_units = Array{SubArray{}}(undef, n_layers)

end


# method that MIGHT work with slices?
function preallocate_minibatch!(mb::Batch_slice, nnw, hp)
    ncols = hp.mb_size_in

    mb.a = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]  
    mb.targets = zeros(nnw.ks[nnw.output_layer], ncols)
    mb.z = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.z_norm  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.delta_z_norm  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.delta_z  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    mb.grad  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    mb.epsilon  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.dropout_random  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
    # mb.dropout_mask_units  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]

    if hp.do_batch_norm
        mb.z_norm  = [zeros(nnw.ks[i],ncols) for i in 1:nnw.output_layer]
        mb.delta_z_norm = deepcopy(mb.epsilon)  # similar z
        mb.delta_z = deepcopy(mb.epsilon)       # similar z
    end

    #    #debug
    # println("size of pre-allocated mb.delta_z_norm $(size(mb.delta_z_norm))")
    # for i in 1:size(mb.delta_z_norm,1)
    #     println("$i size: $(size(mb.delta_z_norm[i]))")
    # end
    # error("that's all folks....")

    if hp.dropout
        mb.dropout_random = deepcopy(mb.epsilon)
        push!(mb.dropout_mask_units,fill(true,(2,2))) # for input layer, not used
        for item in mb.dropout_random[2:end]
            push!(mb.dropout_mask_units,fill(true,size(item)))
        end
    end
end



# cost functions, layer functions: activation, layer functions:  gradient,
#       Classifiers, Optimization, Regularization


###############################################################################
#  cost functions
###############################################################################

function cross_entropy_cost(targets, predictions, n, theta=[], lambda=1.0, reg="", output_layer=3)
    # n is count of all samples in data set--use with regularization term
    # mb_size is count of all samples used in training batch--use with cost
    # these may be equal
    cost = (-1.0 / n) * (dot(targets,log.(predictions .+ 1e-50)) +
        dot((1.0 .- targets), log.(1.0 .- predictions .+ 1e-50)))
        
    @fastmath if reg == "L2"  # set reg="" if not using regularization
        regterm = lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end

function softmax_cost(targets, predictions, n, theta=[], lambda=1.0, reg="", output_layer=3)
    cost = (-1.0 / n) * dot(targets,log.(predictions .+ 1e-50)) 
        
    @fastmath if reg == "L2"  # set reg="" if not using regularization
        regterm = lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end

function mse_cost(targets, predictions, n, theta=[], lambda=1.0, reg="", output_layer=3)
    @fastmath cost = (1.0 / (2.0 * n)) .* sum((targets .- predictions) .^ 2.0)
    @fastmath if reg == "L2"  # set reg="" if not using regularization
        regterm = lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end



###########################################################################
#  layer functions:  activation for feed forward
###########################################################################


# two methods for linear layer units, with bias and without
# function affine!(z, a, theta, bias)  # with bias
#     @inbounds z[:] = theta * a .+ bias
# end

function affine_nobias!(z, a, theta, bias)  # just ignore the bias term
    mul!(z, theta, a)  # crazy fast; no allocations
end

function affine!(z, a, theta, bias)
    # this is really fast with NO allocations!
    mul!(z, theta, a)
    for j = axes(z,2)
        for i = axes(bias, 1)
            z[i,j] += bias[i]
        end
    end
end


function sigmoid!(a::AbstractArray{Float64}, z::AbstractArray{Float64})
    @fastmath a[:] = 1.0 ./ (1.0 .+ exp.(.-z))  
end


function tanh_act!(a::AbstractArray{Float64}, z::AbstractArray{Float64})
    @fastmath a[:] = tanh.(z)
end


function l_relu!(a::AbstractArray{Float64}, z::AbstractArray{Float64}) # leaky relu
    @fastmath a[:] = map(j -> j >= 0.0 ? j : l_relu_neg * j, z)
end


function relu!(a::AbstractArray{Float64}, z::AbstractArray{Float64})
    @fastmath a[:] = max.(z, 0.0)
end


###########################################################################
#  layer functions:  back propagation chain rule
###########################################################################

    # Choice of function determined in setup_functions! in setup_training.jl

    # uses delta_z from the backnorm calculations
    function backprop_weights_nobias!(delta_w, delta_b, delta_z, epsilon, a_prev, n)
        mul!(delta_w, delta_z, a_prev')
        @fastmath delta_w[:] = delta_w .* (1.0 / n)
    end

    # ignores delta_z terms because no batchnorm 
    function backprop_weights!(delta_w, delta_b, delta_z, epsilon, a_prev, n)
        mul!(delta_w, epsilon, a_prev')
        @fastmath delta_w[:] = delta_w .* (1.0 / n)
        @fastmath delta_b[:] = sum(epsilon, dims=2) .* (1.0 / n)
    end



###########################################################################
#  layer functions:  gradient 
###########################################################################


# two methods for gradient of linear layer units:  without bias and with
# not using this yet
function affine_gradient(data, layer)  # no bias
    return data.a[layer-1]'
end


function sigmoid_gradient!(grad::AbstractArray{Float64}, z::AbstractArray{Float64})
    sigmoid!(grad, z)
    @fastmath grad[:] = grad .* (1.0 .- grad)
end


function tanh_act_gradient!(grad::AbstractArray{Float64}, z::AbstractArray{Float64})
    @fastmath grad[:] = 1.0 .- tanh.(z).^2
end


function l_relu_gradient!(grad::AbstractArray{Float64}, z::AbstractArray{Float64})
    grad[:] = map(j -> j > 0.0 ? 1.0 : l_relu_neg, z);
end


function relu_gradient!(grad::AbstractArray{Float64}, z::AbstractArray{Float64})
    # fill!(grad, 0.0)
    @simd for i = eachindex(z)
        if z[i] > 0.0
            grad[i] = 1.0
        else
            grad[i] = 0.0
        end
    end
end


#############################################################################
#  Classifiers
#############################################################################

function softmax!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    expf = similar(a)
    @fastmath expf .= exp.(z .- maximum(z))  # inside maximum: ,dims=1  maximum(z)
    @fastmath a .= expf ./ sum(expf, dims=1)
end


function logistic!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    @fastmath a .= 1.0 ./ (1.0 .+ exp.(.-z))  
end


function regression!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    a[:] = z[:]
end


"""
    function multiclass_pred(preds::AbstractArray{Float64,2})

Converts vector of probabilities of each outcome class to a numeric category.
Returns a 1-dimensional vector of ints with values from 1 to the number of classes.

"""
function multiclass_pred(preds::AbstractArray{Float64,2})
    if size(targets,1) > 1
        predmax = vec(map(x -> x[1], argmax(preds,dims=1)))
    else
        @error("Final targets must contain more than one outcome per example.")
    end
    return predmax
end




##########################################################################
# Optimization
##########################################################################


function dropout_fwd!(dat,hp,hl)  # applied per layer
    @inbounds dat.dropout_random[hl][:] = rand(Float64, size(dat.dropout_random[hl]))
    @inbounds dat.dropout_mask_units[hl][:] = dat.dropout_random[hl] .< hp.droplim[hl]
    # choose activations to remain and scale
    @inbounds dat.a[hl][:] = dat.a[hl] .* (dat.dropout_mask_units[hl] ./ hp.droplim[hl])
end


function dropout_back!(dat, hl)
    @inbounds dat.epsilon[hl][:] = dat.epsilon[hl] .* dat.dropout_mask_units[hl]
end


function step_learn_decay!(hp, ep_i)
    decay_rate = hp.learn_decay[1]
    e_steps = hp.learn_decay[2]
    stepsize = floor(hp.epochs / e_steps)
    if hp.epochs - ep_i < stepsize
        return
    elseif (rem(ep_i,stepsize) == 0.0)
        hp.alpha *= decay_rate
        # hp.alphaovern *= decay_rate
        println("     **** at epoch $ep_i stepping down learning rate to $(hp.alpha)")
    else
        return
    end
end


function momentum!(nnw, hp, t)
    @fastmath for hl = (nnw.output_layer - 1):-1:2  # loop over hidden layers
        @inbounds nnw.delta_v_w[hl] .= hp.b1 .* nnw.delta_v_w[hl] .+ (1.0 - hp.b1) .* nnw.delta_w[hl]  # @inbounds 
        @inbounds nnw.delta_w[hl] .= nnw.delta_v_w[hl]

        if !hp.do_batch_norm  # then we need to do bias term
            @inbounds nnw.delta_v_b[hl] .= hp.b1 .* nnw.delta_v_b[hl] .+ (1.0 - hp.b1) .* nnw.delta_b[hl]  # @inbounds 
            @inbounds nnw.delta_b[hl] .= nnw.delta_v_b[hl]
        end
    end
end


function rmsprop!(nnw, hp, t)
    @fastmath for hl = (nnw.output_layer - 1):-1:2  # loop over hidden layers
        @inbounds nnw.delta_v_w[hl] .= hp.b1 .* nnw.delta_v_w[hl] .+ (1.0 - hp.b1) .* nnw.delta_w[hl].^2   
        @inbounds nnw.delta_w[hl] .=  (nnw.delta_w[hl]  ./   
                              (sqrt.(nnw.delta_v_w[hl]) .+ hp.ltl_eps)  )

        if !hp.do_batch_norm  # then we need to do bias term
            @inbounds nnw.delta_v_b[hl] .= hp.b1 .* nnw.delta_v_b[hl] .+ (1.0 - hp.b1) .* nnw.delta_b[hl].^2   
            @inbounds nnw.delta_b[hl] .= (nnw.delta_b[hl]  ./   
                              (sqrt.(nnw.delta_v_b[hl]) .+ hp.ltl_eps)  )
        end
    end
end


function adam!(nnw, hp, t)
    @fastmath for hl = (nnw.output_layer - 1):-1:2  # loop over hidden layers
        @inbounds nnw.delta_v_w[hl] .= hp.b1 .* nnw.delta_v_w[hl] .+ (1.0 - hp.b1) .* nnw.delta_w[hl]  
        @inbounds nnw.delta_s_w[hl] .= hp.b2 .* nnw.delta_s_w[hl] .+ (1.0 - hp.b2) .* nnw.delta_w[hl].^2   
        @inbounds nnw.delta_w[hl] .= (  (nnw.delta_v_w[hl] ./ (1.0 - hp.b1^t)) ./   
                              sqrt.(nnw.delta_s_w[hl] ./ (1.0 - hp.b2^t) .+ hp.ltl_eps)  )

        if !hp.do_batch_norm  # then we need to do bias term
            @inbounds nnw.delta_v_b[hl] .= hp.b1 .* nnw.delta_v_b[hl] .+ (1.0 - hp.b1) .* nnw.delta_b[hl]   
            @inbounds nnw.delta_s_b[hl] .= hp.b2 .* nnw.delta_s_b[hl] .+ (1.0 - hp.b2) .* nnw.delta_b[hl].^2   
            @inbounds nnw.delta_b[hl] .= (  (nnw.delta_v_b[hl] ./ (1.0 - hp.b1^t)) ./   
                              sqrt.(nnw.delta_s_b[hl] ./ (1.0 - hp.b2^t) .+ hp.ltl_eps) )  
        end
    end
end


##########################################################################
# Regularization
##########################################################################

function maxnorm_reg!(nnw, hp, hl)    # (theta, maxnorm_lim)
    theta = nnw.theta[hl]
    maxnorm_lim = hp.maxnorm_lim[hl]
    for i in 1:size(theta,1)  
        # row i of theta contains weights for output of unit i in current layer
        # column values in row i multiplied times input values from next lower layer activations
        norm_of_unit = norm(theta[i,:]) 
        if norm_of_unit > maxnorm_lim
            theta[i,:] .= theta[i,:] .* (maxnorm_lim / norm_of_unit)
        end
    end
end

function l2_reg!(nnw, hp, hl)
    @inbounds nnw.theta[hl] .= nnw.theta[hl] .+ (hp.lambda / hp.mb_size .* nnw.theta[hl])
end

function l1_reg!(nnw, hp, hl)
    @inbounds nnw.theta[hl] .= nnw.theta[hl] .+ (hp.lambda / hp.mb_size .* sign.(nnw.theta[hl]))
end


############## noop stub
function noop(args...)
end
# cost functions, layer functions: activation, layer functions:  gradient,
#       Classifiers, Optimization, Regularization


###############################################################################
#  cost functions
###############################################################################

function cross_entropy_cost(targets, predictions, n, theta=[], lambda=0.1, reg="", output_layer=3)
    # n is count of all samples in data set--use with regularization term
    # mb_size is count of all samples used in training batch--use with cost
    # these may be equal
    cost = (-1.0 / n) * (dot(targets,log.(max.(predictions, 1e-20))) +
        dot((1.0 .- targets), log.(max.(1.0 .- predictions, 1e-20))))
        
    @fastmath if reg == "L2"  # set reg="" if not using regularization
        regterm = lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end

function softmax_cost(targets, predictions, n, theta=[], lambda=0.1, reg="", output_layer=3)
    # this is the negative log likelihood cost for a multi-category output layer
    cost = (-1.0 / n) * dot(targets,log.(max.(predictions, 1e-20))) 
        
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

# testing behavior of cross_entropy_cost
function cost_target_one(targets, predictions, n, theta=[], lambda=1.0, reg="", output_layer=3)
    cost = (-1.0 / n) * dot(targets,log.(predictions .+ 1e-50)) 
end

function cost_target_zero(targets, predictions, n, theta=[], lambda=1.0, reg="", output_layer=3)
   cost = (-1.0 / n) * dot((1.0 .- targets), log.(1.0 .- predictions .+ 1e-50))
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
#  layer functions for back propagation
###########################################################################

    # Choice of function determined in setup_functions! in setup_training.jl

    function backprop_classify!(epsilon, preds, targets)
        epsilon[:] = preds .- targets  
    end


    function inbound_epsilon!(epsilon, theta_above, eps_lr_above)
        mul!(epsilon, theta_above', eps_lr_above)
    end


    function current_lr_epsilon!(epsilon, grad)
        @inbounds epsilon[:] = epsilon .* grad
    end


    # uses epsilon from the batchnorm_back calculations
    function backprop_weights_nobias!(delta_th, delta_b, epsilon, a_prev, n)
        mul!(delta_th, epsilon, a_prev')
        @fastmath delta_th[:] = delta_th .* (1.0 / n)
    end

    function backprop_weights!(delta_th, delta_b, epsilon, a_prev, n)
        mul!(delta_th, epsilon, a_prev')

        @fastmath delta_th[:] = delta_th .* (1.0 / n)
        @fastmath delta_b[:] = sum(epsilon, dims=2) ./ n
    end



###########################################################################
#  layer functions:  gradient 
###########################################################################


# two methods for gradient of linear layer units:  without bias and with
# not using this yet
function affine_gradient!(data, layer)  # no bias
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
    @simd for i = eachindex(z)
        @inbounds if z[i] > 0.0
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
    # z[:] = clamp.(z,-300.0, 300.0) # prevent NaNs in softmax with crazy learning rate
    expf = similar(a)
    maxz = maximum(z)
    @fastmath expf .= exp.(z .- maxz)  # inside maximum: ,dims=1  maximum(z)
    @fastmath a .= expf ./  sum(expf, dims=1)  # 
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


function dropout_fwd!(dat, hp, nnw, hl)  # applied per layer

    @inbounds nnw.dropout_mask_units[hl][:] = Bool.(rand( Bernoulli( hp.droplim[hl] ), size(nnw.dropout_mask_units[hl], 1)))
    # choose activations to remain and scale
    @inbounds dat.a[hl][:] = dat.a[hl] .* (nnw.dropout_mask_units[hl] ./ hp.droplim[hl])
end


function dropout_back!(dat, nnw, hp, hl)
    @inbounds dat.epsilon[hl][:] = dat.epsilon[hl] .* (nnw.dropout_mask_units[hl])
end


function step_learn_decay!(hp, ep_i)
    decay_rate = hp.learn_decay[1]
    e_steps = hp.learn_decay[2]
    stepsize = floor(hp.epochs / e_steps)
    if hp.epochs - ep_i < stepsize
        return
    elseif (rem(ep_i,stepsize) == 0.0)
        hp.alphamod *= decay_rate
        println("     **** at epoch $ep_i stepping down learning rate to $(hp.alphamod)")
    else
        return
    end
end


function momentum!(nnw, hp, bn, t)
    @fastmath for hl = (nnw.output_layer - 1):-1:2  # loop over hidden layers
        @inbounds nnw.delta_v_th[hl][:] = hp.b1 .* nnw.delta_v_th[hl] .+ (1.0 - hp.b1) .* nnw.delta_th[hl]  # @inbounds 
        @inbounds nnw.delta_th[hl][:] = nnw.delta_v_th[hl]

        if !hp.do_batch_norm  # no batchnorm so we need to do bias term
            @inbounds nnw.delta_v_b[hl][:] = hp.b1 .* nnw.delta_v_b[hl] .+ (1.0 - hp.b1) .* nnw.delta_b[hl]  # @inbounds 
            @inbounds nnw.delta_b[hl][:] = nnw.delta_v_b[hl]
        elseif hp.opt_batch_norm # yes, doing batchnorm, but don't use optimization
            @inbounds bn.delta_v_gam[hl][:] = hp.b1 .* bn.delta_v_gam[hl] .+ (1.0 - hp.b1) .* bn.delta_gam[hl]  # @inbounds 
            @inbounds bn.delta_gam[hl][:] = bn.delta_v_gam[hl]
            @inbounds bn.delta_v_bet[hl][:] = hp.b1 .* bn.delta_v_bet[hl] .+ (1.0 - hp.b1) .* bn.delta_bet[hl]  # @inbounds 
            @inbounds bn.delta_bet[hl][:] = bn.delta_v_bet[hl]
        end
    end
end


function rmsprop!(nnw, hp, bn, t)
    @fastmath for hl = (nnw.output_layer - 1):-1:2  # loop over hidden layers
        @inbounds nnw.delta_v_th[hl][:] = hp.b1 .* nnw.delta_v_th[hl] .+ (1.0 - hp.b1) .* nnw.delta_th[hl].^2   
        @inbounds nnw.delta_th[hl][:] =  nnw.delta_th[hl] ./  (sqrt.(nnw.delta_v_th[hl]) .+ hp.ltl_eps)

        if !hp.do_batch_norm  # then we need to do bias term
            @inbounds nnw.delta_v_b[hl][:] = hp.b1 .* nnw.delta_v_b[hl] .+ (1.0 - hp.b1) .* nnw.delta_b[hl].^2   
            @inbounds nnw.delta_b[hl][:] = nnw.delta_b[hl] ./ (sqrt.(nnw.delta_v_b[hl]) .+ hp.ltl_eps)
        elseif hp.opt_batch_norm # yes, doing batchnorm, but don't use optimization
            @inbounds bn.delta_v_gam[hl][:] = hp.b1 .* bn.delta_v_gam[hl] .+ (1.0 - hp.b1) .* bn.delta_gam[hl].^2   
            @inbounds bn.delta_gam[hl][:] = bn.delta_gam[hl] ./ (sqrt.(bn.delta_v_gam[hl]) .+ hp.ltl_eps)
            @inbounds bn.delta_v_bet[hl][:] = hp.b1 .* bn.delta_v_bet[hl] .+ (1.0 - hp.b1) .* bn.delta_bet[hl].^2   
            @inbounds bn.delta_bet[hl][:] = bn.delta_bet[hl] ./ (sqrt.(bn.delta_v_bet[hl]) .+ hp.ltl_eps)
        end
    end
end


function adam!(nnw, hp, bn, t)
    @fastmath for hl = (nnw.output_layer - 1):-1:2  # loop over hidden layers
        adam_helper!(nnw.delta_v_th[hl], nnw.delta_s_th[hl], nnw.delta_th[hl], hp, t)

        if !hp.do_batch_norm  # then we need to do bias term
            adam_helper!(nnw.delta_v_b[hl], nnw.delta_s_b[hl], nnw.delta_b[hl], hp, t)
        elseif hp.opt_batch_norm # yes, doing batchnorm, but don't use optimization
            adam_helper!(bn.delta_v_gam[hl], bn.delta_s_gam[hl], bn.delta_gam[hl], hp, t)
            adam_helper!(bn.delta_v_bet[hl], bn.delta_s_bet[hl], bn.delta_bet[hl], hp, t)            
        end
    end
end

function adam_helper!(v_w, s_w, w, hp,t)
    @inbounds v_w[:] = hp.b1 .* v_w .+ (1.0 - hp.b1) .* w  
    @inbounds s_w[:] = hp.b2 .* s_w .+ (1.0 - hp.b2) .* w.^2   
    @inbounds w[:] = (v_w ./ (1.0 - hp.b1^t)) ./ sqrt.(s_w ./ (1.0 - hp.b2^t) .+ hp.ltl_eps)
end

##########################################################################
# Regularization
##########################################################################

# batch normalization
# method for training that updates running averages of mu and std
function batch_norm_fwd!(dat::Union{Model_data, Batch_view}, bn::Batch_norm_params, hp::Hyper_parameters, hl::Int)
!hp.quiet && println("batch_norm_fwd!(dat, bn, hp, hl)")

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

# method for prediction using running average of mu and std
function batch_norm_fwd!(dat::Union{Model_data, Batch_view}, bn::Batch_norm_params, hp::Hyper_parameters, hl::Int, notrain)
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
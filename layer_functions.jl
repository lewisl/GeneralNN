# All layer functions except cost function are used in the training loop
#    Training loop runs 3 sub-loops:
#       feedfwd!
#       backprop!
#       update_parameters! which includes optimization and regularization

# each with 2 methods:  "outer" method visible in the loop accepts all loop inputs
    # "inner" method is the function that receives needed arguments to do the calculations


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
        @inbounds regterm = lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end

function softmax_cost(targets, predictions, n, theta=[], lambda=0.1, reg="", output_layer=3)
    # this is the negative log likelihood cost for a multi-category output layer
    cost = (-1.0 / n) * dot(targets,log.(max.(predictions, 1e-20))) 
        
    @fastmath if reg == "L2"  # set reg="" if not using regularization
        @inbounds regterm = lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
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
#  layer functions:  activation used in feedfwd!
###########################################################################

# feedfwd arguments
# (dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
#        bn::Batch_norm_params, lr::Int, dotrain)
#

# two methods for linear layer units, with bias and without
# function affine!(z, a, theta, bias)  # with bias
#     @inbounds z[:] = theta * a .+ bias
# end


# function affine_nobias!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
#         bn::Batch_norm_params, lr::Int, dotrain)
# function affine_nobias!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
#         bn::Batch_norm_params, lr::Int, dotrain)
function affine_nobias!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int, dotrain)
    affine_nobias!(dat.z[lr], dat.a[lr-1], nnw.theta[lr])
end


function affine_nobias!(z, a, theta)  # just ignore the bias term
    mul!(z, theta, a)  # crazy fast; no allocations
end


function affine!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int, dotrain)
    affine!(dat.z[lr], dat.a[lr-1], nnw.theta[lr], nnw.bias[lr])
end

function affine!(z, a, theta, bias)
    # this is really fast with NO allocations!
    mul!(z, theta, a)
    @simd for j = axes(z,2)
        @simd for i = axes(bias, 1)
            @inbounds z[i,j] += bias[i]
        end
    end
end


function sigmoid!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int, dotrain)
    sigmoid!(dat.a[lr], dat.z[lr])
end

function sigmoid!(a::AbstractArray{Float64}, z::AbstractArray{Float64})
    @fastmath a[:] = 1.0 ./ (1.0 .+ exp.(.-z))  
end


function tanh_act!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int, dotrain)
    tanh_act!(dat.a[lr], dat.z[lr])
end

function tanh_act!(a::AbstractArray{Float64}, z::AbstractArray{Float64})
    @fastmath a[:] = tanh.(z)
end


function l_relu!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int, dotrain)
    l_relu!(dat.a[lr], dat.z[lr])
end

function l_relu!(a::AbstractArray{Float64}, z::AbstractArray{Float64}) # leaky relu
    @fastmath a[:] = map(j -> j >= 0.0 ? j : l_relu_neg * j, z)
end


function relu!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int, dotrain)
    relu!(dat.a[lr], dat.z[lr])
end

function relu!(a::AbstractArray{Float64}, z::AbstractArray{Float64})
    @fastmath a[:] = max.(z, 0.0)
end


#############################################################################
#  Classifiers used in feedfwd! loop
#############################################################################


function softmax!(dat, nnw, hp, bn, lr, dotrain)
    softmax!(dat.a[lr], dat.z[lr])
end

function softmax!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    # z[:] = clamp.(z,-300.0, 300.0) # prevent NaNs in softmax with crazy learning rate
    expf = similar(a)
    maxz = maximum(z)
    @fastmath expf .= exp.(z .- maxz)  # inside maximum: ,dims=1  maximum(z)
    @fastmath a .= expf ./  sum(expf, dims=1)  # 
end


function logistic!(dat, nnw, hp, bn, lr, dotrain)
    logistic!(dat.a[lr], dat.z[lr])
end

function logistic!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    @fastmath a .= 1.0 ./ (1.0 .+ exp.(.-z))  
end


function regression!(dat, nnw, hp, bn, lr, dotrain)
    regression!(dat.a[lr], dat.z[lr])
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



###########################################################################
#  layer functions in backprop!
###########################################################################

# backprop loop arguments:
# (dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
#     bn::Batch_norm_params, lr::Int)
#


function backprop_classify!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    # this will only ever be called when hl is the output layer: do we need to force it?
    backprop_classify!(dat.epsilon[lr], dat.a[lr], dat.targets)
end

function backprop_classify!(epsilon, preds, targets)
    @inbounds epsilon[:] = preds .- targets  
end


function inbound_epsilon!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    inbound_epsilon!(dat.epsilon[lr], nnw.theta[lr+1], dat.epsilon[lr+1])
end

function inbound_epsilon!(epsilon, theta_above, eps_above)
    mul!(epsilon, theta_above', eps_above)
end


function current_lr_epsilon!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    current_lr_epsilon!(dat.epsilon[lr], dat.grad[lr])
end

function current_lr_epsilon!(epsilon, grad)
    @fastmath @inbounds epsilon[:] = epsilon .* grad
end


function backprop_weights_nobias!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    backprop_weights_nobias!(nnw.delta_th[lr], dat.epsilon[lr], dat.a[lr-1], hp.mb_size)
end

# uses epsilon from the batchnorm_back calculations
function backprop_weights_nobias!(delta_th, epsilon, a_below, n)
    mul!(delta_th, epsilon, a_below')
    @fastmath delta_th[:] = delta_th .* (1.0 / n)
end


function backprop_weights!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    backprop_weights!(nnw.delta_th[lr], nnw.delta_b[lr], dat.epsilon[lr], dat.a[lr-1], hp.mb_size)
end

function backprop_weights!(delta_th, delta_b, epsilon, a_below, n)
    mul!(delta_th, epsilon, a_below')

    @fastmath delta_th[:] = delta_th .* (1.0 / n)
    @fastmath delta_b[:] = sum(epsilon, dims=2) ./ n
end


###########################################################################
#  layer functions in backprop!  gradient 
###########################################################################

# two methods for gradient of linear layer units:  without bias and with
# not using this yet
function affine_gradient!(dat, lr)  # no bias
    return dat.a[layer-1]'
end


function sigmoid_gradient!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    sigmoid_gradient!(dat.grad[lr], dat.z[lr])
end

function sigmoid_gradient!(grad::AbstractArray{Float64}, z::AbstractArray{Float64})
    sigmoid!(grad, z)
    @fastmath grad[:] = grad .* (1.0 .- grad)
end


function tanh_act_gradient!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    tanh_act_gradient!(dat.grad[lr], dat.z[lr])
end

function tanh_act_gradient!(grad::AbstractArray{Float64}, z::AbstractArray{Float64})
    @fastmath grad[:] = 1.0 .- tanh.(z).^2
end


function l_relu_gradient!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    l_relu_gradient!(dat.grad[lr], dat.z[lr])
end

function l_relu_gradient!(grad::AbstractArray{Float64}, z::AbstractArray{Float64})
    grad[:] = map(j -> j > 0.0 ? 1.0 : l_relu_neg, z);
end


function relu_gradient!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    relu_gradient!(dat.grad[lr], dat.z[lr])
end

function relu_gradient!(grad::T_array_subarray, z::T_array_subarray)
    @simd for i = eachindex(z)
        @inbounds if z[i] > 0.0
            grad[i] = 1.0
        else
            grad[i] = 0.0
        end
    end
end


###########################################################################
#  layer functions in update_parameters!
###########################################################################

# These can be built with a single function that takes all feedfwd arguments
#    or 2 methods:  one accepts all arguments then passes only needed arguments to a 2nd method

# arguments for functions in update_parameters
# (nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, lr::Int, t::Int)
#

# function update_wgts!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, lr::Int, t::Int)
#     update_wgts!(nnw.theta[lr], nnw.bias[lr], hp.alphamod, nnw.delta_th[lr], nnw.delta_b[lr])
# end

# method for a single layer--caller indexes the array of arrays to pass single layer array
# (nnw.theta[hl], nnw.bias[hl], hp.alphamod, nnw.delta_th[hl], nnw.delta_b[hl])
# function update_wgts!(theta, bias, alpha, delta_th, delta_b)   # could differentiate with method dispatch
#     @fastmath @inbounds theta[:] = theta .- (alpha .* delta_th)
#     @fastmath @inbounds bias[:] .= bias .- (alpha .* delta_b)
# end
function update_wgts!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, lr::Int, t::Int)   
    theta = nnw.theta[lr]
    bias = nnw.bias[lr]
    alpha = hp.alphamod
    delta_th = nnw.delta_th[lr]
    delta_b = nnw.delta_b[lr]

    @fastmath @inbounds theta[:] = theta .- (alpha .* delta_th)
    @fastmath @inbounds bias[:] .= bias .- (alpha .* delta_b)
end


function update_wgts_nobias!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, lr::Int, t::Int)
    update_wgts_nobias!(nnw.theta[lr], hp.alphamod, nnw.delta_th[lr])
end

# method for a single layer--caller indexes the array of arrays to pass single layer array
function update_wgts_nobias!(theta, alpha, delta_th) # (nnw.theta[hl], hp.alphamod, nnw.delta_th[hl])
    @fastmath @inbounds theta[:] = theta .- (alpha .* delta_th)
end


function update_batch_norm!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, lr::Int, t::Int)
    update_batch_norm!(bn.gam[lr], bn.bet[lr], hp.alphamod, bn.delta_gam[lr], bn.delta_bet[lr])
end

# method for a single layer--caller indexes the array of arrays to pass single layer array
# (bn.gam[hl], bn.bet[hl], hp.alphamod, bn.delta_gam[hl], bn.delta_bet[hl])
function update_batch_norm!(gam, bet, alpha, delta_gam, delta_bet)
    @fastmath @inbounds gam[:] = gam .- (alpha .* delta_gam)
    @fastmath @inbounds bet[:] = bet .- (alpha .* delta_bet)
end


##########################################################################
# Optimization
##########################################################################

function dropout_fwd!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int, dotrain::Bool)
    dropout_fwd!(dat.a[lr], nnw.dropout_mask[lr], hp.droplim[lr], dotrain)
end

function dropout_fwd!(a, dropout_mask, droplim, dotrain=true)  # applied on single layer
    dotrain && begin
        @inbounds dropout_mask[:] = Bool.(rand( Bernoulli( droplim ), size(dropout_mask, 1)))
        # choose activations to remain and scale
        @inbounds a[:] = a .* (dropout_mask ./ droplim) # "inverted" dropout
    end
end


function dropout_back!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    dropout_back!(dat.epsilon[lr], nnw.dropout_mask[lr], hp.droplim[lr])
end

function dropout_back!(epsilon, dropout_mask, droplim)
    @inbounds epsilon[:] = epsilon .* (dropout_mask ./ droplim) # "inverted" dropout
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


# function momentum!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, t::Int)
#     momentum!(nnw, hp, bn, hl, t)
# end

# method for multiple layers in a loop
function momentum!(nnw, hp, bn, t)
    @fastmath @simd for hl = (nnw.output_layer - 1):-1:2  # loop over hidden layers
        momentum!(nnw, hp, bn, hl, t)  # this will get inlined by the compiler
    end
end


# method for a single layer
function momentum!(nnw, hp, bn, hl, t)
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


# function rmsprop!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, t::Int)
#     rmsprop!(nnw, hp, bn, hl, t)
# end

# method for looping over hidden layers
function rmsprop!(nnw, hp, bn, t)
    @fastmath @simd for hl = (nnw.output_layer - 1):-1:2  # loop over hidden layers
        rmsprop!(nnw, hp, bn, hl, t)
    end
end

# method for a single layer
function rmsprop!(nnw, hp, bn, hl, t)
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


# function adam!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, t::Int)
#     adam!(nnw, hp, bn, hl, t)  # method for single layer
# end

# method that loops over hidden layers
function adam!(nnw, hp, bn, t)
    @fastmath @simd for hl = (nnw.output_layer - 1):-1:2  # loop over hidden layers
        adam!(nnw, hp, bn, hl, t)
    end
end

# method for a single layer
function adam!(nnw, hp, bn, hl, t)
    adam_helper!(nnw.delta_v_th[hl], nnw.delta_s_th[hl], nnw.delta_th[hl], hp, t)

    if !hp.do_batch_norm  # then we need to do bias term
        adam_helper!(nnw.delta_v_b[hl], nnw.delta_s_b[hl], nnw.delta_b[hl], hp, t)
    elseif hp.opt_batch_norm # yes, doing batchnorm, and optimization of batch_norm params
        adam_helper!(bn.delta_v_gam[hl], bn.delta_s_gam[hl], bn.delta_gam[hl], hp, t)
        adam_helper!(bn.delta_v_bet[hl], bn.delta_s_bet[hl], bn.delta_bet[hl], hp, t)            
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

# TODO:  if we changed the order or input parameters we would not need the outer method
function batch_norm_fwd!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int, dotrain::Bool)
    batch_norm_fwd!(dat, bn, hp, lr, dotrain)
end

function batch_norm_fwd!(dat::Union{Model_data, Batch_view}, bn::Batch_norm_params, 
    hp::Hyper_parameters, hl::Int, dotrain=true)
    !hp.quiet && println("batch_norm_fwd!(dat, bn, hp, hl)")

    dotrain && @inbounds begin
            bn.mu[hl][:] = mean(dat.z[hl], dims=2)          # use in backprop
            bn.stddev[hl][:] = std(dat.z[hl], dims=2)
            dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu[hl]) ./ (bn.stddev[hl] .+ hp.ltl_eps) # normalized: often xhat or zhat  
            dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: often called y 
            bn.mu_run[hl][:] = (  bn.mu_run[hl][1] == 0.0 ? bn.mu[hl] :  
                0.95 .* bn.mu_run[hl] .+ 0.05 .* bn.mu[hl]  )
            bn.std_run[hl][:] = (  bn.std_run[hl][1] == 0.0 ? bn.stddev[hl] :  
                0.95 .* bn.std_run[hl] + 0.05 .* bn.stddev[hl]  )
            return
    end

    # else, e.g.--dotrain == false
    @inbounds dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu_run[hl]) ./ (bn.std_run[hl] .+ hp.ltl_eps) # normalized: aka xhat or zhat 
    @inbounds dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: often called y 

end

# # method for prediction using running average of mu and std
# function batch_norm_fwd!(dat::Union{Model_data, Batch_view}, bn::Batch_norm_params, hp::Hyper_parameters, hl::Int, notrain)
# !hp.quiet && println("batch_norm_fwd_predict!(hp, bn, dat, hl)")

#     @inbounds dat.z_norm[hl][:] = (dat.z[hl] .- bn.mu_run[hl]) ./ (bn.std_run[hl] .+ hp.ltl_eps) # normalized: aka xhat or zhat 
#     @inbounds dat.z[hl][:] = dat.z_norm[hl] .* bn.gam[hl] .+ bn.bet[hl]  # shift & scale: often called y 
# end


# TODO if we changed the order of parameters we would not need the outer method
function batch_norm_back!(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, lr::Int)
    batch_norm_back!(nnw, dat, bn, lr, hp)
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


function maxnorm_reg!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, lr::Int, 
        t::Int)
    maxnorm_reg!(nnw.theta, hp, lr)
end

function maxnorm_reg!(theta, hp, hl)    
    theta = theta[hl]
    maxnorm_lim = hp.maxnorm_lim[hl]
    @simd for i in 1:size(theta,1)  
        # row i of theta contains weights for output of unit i in current layer
        # column values in row i multiplied times input values from next lower layer activations
        norm_of_unit = norm(theta[i,:]) 
        if norm_of_unit > maxnorm_lim
            @inbounds theta[i,:] .= theta[i,:] .* (maxnorm_lim / norm_of_unit)
        end
    end
end


function l2_reg!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, lr::Int, 
        t::Int)
    l2_reg!(nnw.theta, hp, lr)
end

function l2_reg!(theta, hp, hl)
    @inbounds theta[hl][:] = theta[hl] .+ (hp.lambda / hp.mb_size .* theta[hl])
end


function l1_reg!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, lr::Int, 
        t::Int)
    l1_reg!(nnw.theta, hp, lr)
end

function l1_reg!(theta, hp, hl)
    @inbounds theta[hl][:] = theta[hl] .+ (hp.lambda / hp.mb_size .* sign.(theta[hl]))
end


############## noop stub  
function noop(args...)
end
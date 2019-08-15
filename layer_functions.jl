# cost functions, layer functions: activation, layer functions:  gradient,
#       Classifiers, Optimization, Regularization


###############################################################################
#  cost functions
###############################################################################

function cross_entropy_cost(targets, predictions, n, theta, hp, output_layer)
    # n is count of all samples in data set--use with regularization term
    # mb_size is count of all samples used in training batch--use with cost
    # these may be equal
    cost = (-1.0 / n) * (dot(targets,log.(predictions .+ 1e-50)) +
        dot((1.0 .- targets), log.(1.0 .- predictions .+ 1e-50)))

    @fastmath if hp.reg == "L2"  # set reg="" if not using regularization
        # regterm = hp.lambda/(2.0 * n) .* sum([sum(th .* th) for th in theta[2:output_layer]])
        regterm = hp.lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end


function mse_cost(targets, predictions, n, theta, hp, output_layer)
    @fastmath cost = (1.0 / (2.0 * n)) .* sum((targets .- predictions) .^ 2.0)
    @fastmath if hp.reg == "L2"  # set reg="" if not using regularization
        regterm = hp.lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end



###########################################################################
#  layer functions:  activation 
###########################################################################


# two methods for linear layer units, with bias and without
# function affine!(z, a, theta, bias)  # with bias
#     @inbounds z[:] = theta * a .+ bias
# end

function affine!(z, a, theta)  # no bias
    mul!(z, theta, a)
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


function sigmoid!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    @fastmath a[:] = 1.0 ./ (1.0 .+ exp.(.-z))  
end

function tanh_act!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    @fastmath a[:] = tanh.(z)
end

function l_relu!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2}) # leaky relu
    @fastmath a[:] = map(j -> j >= 0.0 ? j : l_relu_neg * j, z)
end


function relu!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    a[:] = max.(z, 0.0)
end


###########################################################################
#  layer functions:  gradient 
###########################################################################


# two methods for gradient of linear layer units:  without bias and with
# not using this yet
function affine_gradient(data, layer)  # no bias
    return data.a[layer-1]'
end


function sigmoid_gradient!(grad::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    sigmoid!(z, grad)
    @fastmath grad[:] = grad .* (1.0 .- grad)
end


function tanh_act_gradient!(grad::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    @fastmath grad[:] = 1.0 .- tanh.(z).^2
end


function l_relu_gradient!(grad::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    grad[:] = map(j -> j > 0.0 ? 1.0 : l_relu_neg, z);
end


function relu_gradient!(grad::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    grad[:] .= 0.0
    @simd for i = 1:length(z)
        if z[i] > 0.0
            grad[i] = 1.0
        end
    end
end


#############################################################################
#  Classifiers
#############################################################################

function softmax!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    expf = similar(a)
    @fastmath expf .= exp.(z .- maximum(z,dims=1))
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


function dropout!(dat,hp,hl)  # applied per layer
    @inbounds dat.dropout_random[hl][:] = rand(Float64, size(dat.dropout_random[hl]))
    @inbounds dat.dropout_mask_units[hl][:] = dat.dropout_random[hl] .< hp.droplim[hl]
    # choose activations to remain and scale
    @inbounds dat.a[hl][:] = dat.a[hl] .* (dat.dropout_mask_units[hl] ./ hp.droplim[hl])
end


function step_learn_decay!(hp, ep_i)
    decay_rate = hp.learn_decay[1]
    e_steps = hp.learn_decay[2]
    stepsize = floor(hp.epochs / e_steps)
    if hp.epochs - ep_i < stepsize
        return
    elseif (rem(ep_i,stepsize) == 0.0)
        hp.alpha *= decay_rate
        hp.alphaovermb *= decay_rate
        println("     **** at epoch $ep_i stepping down learning rate to $(hp.alpha)")
    else
        return
    end
end


function momentum!(nnp, hp, t)
    @fastmath for hl = (nnp.output_layer - 1):-1:2  # loop over hidden layers
        @inbounds nnp.delta_v_w[hl] .= hp.b1 .* nnp.delta_v_w[hl] .+ (1.0 - hp.b1) .* nnp.delta_w[hl]  # @inbounds 
        @inbounds nnp.delta_w[hl] .= nnp.delta_v_w[hl]

        if !hp.do_batch_norm  # then we need to do bias term
            @inbounds nnp.delta_v_b[hl] .= hp.b1 .* nnp.delta_v_b[hl] .+ (1.0 - hp.b1) .* nnp.delta_b[hl]  # @inbounds 
            @inbounds nnp.delta_b[hl] .= nnp.delta_v_b[hl]
        end
    end
end


function rmsprop!(nnp, hp, t)
    @fastmath for hl = (nnp.output_layer - 1):-1:2  # loop over hidden layers
        @inbounds nnp.delta_v_w[hl] .= hp.b1 .* nnp.delta_v_w[hl] .+ (1.0 - hp.b1) .* nnp.delta_w[hl].^2   
        @inbounds nnp.delta_w[hl] .=  (nnp.delta_w[hl]  ./   
                              (sqrt.(nnp.delta_v_w[hl]) .+ hp.ltl_eps)  )

        if !hp.do_batch_norm  # then we need to do bias term
            @inbounds nnp.delta_v_b[hl] .= hp.b1 .* nnp.delta_v_b[hl] .+ (1.0 - hp.b1) .* nnp.delta_b[hl].^2   
            @inbounds nnp.delta_b[hl] .= (nnp.delta_b[hl]  ./   
                              (sqrt.(nnp.delta_v_b[hl]) .+ hp.ltl_eps)  )
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


function no_optimization(nnp, hp, t)
end


##########################################################################
# Regularization
##########################################################################

function maxnorm_reg!(nnp, hp, hl)    # (theta, maxnorm_lim)
    theta = nnp.theta[hl]
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

function l2_reg!(nnp, hp, hl)
    @inbounds nnp.theta[hl] .= nnp.theta[hl] .+ (hp.alphaovermb .* (hp.lambda .* nnp.theta[hl]))
end

function l1_reg!(nnp, hp, hl)
    @inbounds nnp.theta[hl] .= nnp.theta[hl] .+ (hp.alphaovermb .* (hp.lambda .* sign.(nnp.theta[hl])))
end

function no_reg(nnp, hp, hl)
end
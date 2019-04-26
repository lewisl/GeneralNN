


function cross_entropy_cost(targets, predictions, n, theta, hp, output_layer)
    # n is count of all samples in data set--use with regularization term
    # mb_size is count of all samples used in training batch--use with cost
    # these may be equal
    cost = (-1.0 / n) * (dot(targets,log.(predictions .+ 1e-50)) +
        dot((1.0 .- targets), log.(1.0 .- predictions .+ 1e-50)))

    # debug cost calc
    # if isnan(cost)
    #    println("predictions <= 0? ",any(predictions .== 0.0))
    #    println("log 1 - predictions? ", any(isinf.(log.(1.0 .- predictions))))
    #    error("problem with cost function")
    # end

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


###########################################################################
#  layer functions:  activation and gradients for units of different types
###########################################################################

# two methods for linear layer units, with bias and without
function affine!(z, a, theta, bias)  # with bias
    @fastmath z[:] = theta * a .+ bias
end


function affine!(z, a, theta)  # no bias
    @fastmath z[:] = theta * a
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

#############################################################################
# Classifiers
#############################################################################

function softmax!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    expf = similar(a)
    @fastmath expf[:] = exp.(z .- maximum(z,dims=1))
    @fastmath a[:] = expf ./ sum(expf, dims=1)
end


function logistic!(a::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    @fastmath a[:] = 1.0 ./ (1.0 .+ exp.(.-z))  
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


# function relu_gradient!(grad::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
#     grad[:] = map(j -> j > 0.0 ? 1.0 : 0.0, z);
# end

# optimized? relu_gradient!
function relu_gradient!(grad::AbstractArray{Float64,2}, z::AbstractArray{Float64,2})
    grad[:] .= 0.0
    for i = 1:length(z)
        if z[i] > 0.0
            grad[i] = 1.0
        end
    end
end


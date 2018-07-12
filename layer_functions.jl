


function cross_entropy_cost(targets, predictions, n, theta, hp, output_layer)
    # n is count of all samples in data set--use with regularization term
    # mb_size is count of all samples used in training batch--use with cost
    # these may be equal
    cost = (-1.0 / n) * (dot(targets,log.(predictions .+ 1e-50)) +
        dot((1.0 .- targets), log.(1.0 .- predictions .+ 1e-50)))

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
    cost = (1.0 / (2.0 * n)) .* sum((targets .- predictions) .^ 2.0)
    @fastmath if hp.reg == "L2"  # set reg="" if not using regularization
        regterm = hp.lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end


# not using yet
function mse_grad!(mb, layer) # only for output layer using mse_cost
    # do we really need this?

    mb.grad[layer] = mb.a[layer-1]

end


function momentum!(tp, hp, t)
    @fastmath for hl = (tp.output_layer - 1):-1:2  # loop over hidden layers
        tp.delta_v_w[hl] .= hp.b1 .* tp.delta_v_w[hl] .+ (1.0 - hp.b1) .* tp.delta_w[hl]  # @inbounds 
        tp.delta_w[hl] .= tp.delta_v_w[hl]

        if !hp.do_batch_norm  # then we need to do bias term
            tp.delta_v_b[hl] .= hp.b1 .* tp.delta_v_b[hl] .+ (1.0 - hp.b1) .* tp.delta_b[hl]  # @inbounds 
            tp.delta_b[hl] .= tp.delta_v_b[hl]
        end
    end
end


function adam!(tp, hp, t)
    @fastmath for hl = (tp.output_layer - 1):-1:2  # loop over hidden layers
        tp.delta_v_w[hl] .= hp.b1 .* tp.delta_v_w[hl] .+ (1.0 - hp.b1) .* tp.delta_w[hl]  # @inbounds 
        tp.delta_s_w[hl] .= hp.b2 .* tp.delta_s_w[hl] .+ (1.0 - hp.b2) .* tp.delta_w[hl].^2  # @inbounds 
        tp.delta_w[hl] .= (  (tp.delta_v_w[hl] ./ (1.0 - hp.b1^t)) ./  # @inbounds 
                              (sqrt.(tp.delta_s_w[hl] ./ (1.0 - hp.b2^t)) + hp.ltl_eps)  )

        if !hp.do_batch_norm  # then we need to do bias term
            tp.delta_v_b[hl] .= hp.b1 .* tp.delta_v_b[hl] .+ (1.0 - hp.b1) .* tp.delta_b[hl]  # @inbounds 
            tp.delta_s_b[hl] .= hp.b2 .* tp.delta_s_b[hl] .+ (1.0 - hp.b2) .* tp.delta_b[hl].^2  # @inbounds 
            tp.delta_b[hl] .= (  (tp.delta_v_b[hl] ./ (1.0 - hp.b1^t)) ./  # @inbounds 
                              (sqrt.(tp.delta_s_b[hl] ./ (1.0 - hp.b2^t)) + hp.ltl_eps)  )
        end
    end
end


function no_optimization(tp, hp, t)
end


function dropout!(dat,hp,hl)
    dat.drop_ran_w[hl][:] = rand(size(dat.drop_ran_w[hl]))
    dat.drop_filt_w[hl][:] = dat.drop_ran_w[hl] .< hp.droplim[hl]
    dat.a[hl][:] = dat.a[hl] .* dat.drop_filt_w[hl]
    dat.a[hl][:] = dat.a[hl] ./ hp.droplim[hl]
end


function step_lrn_decay!(hp, ep_i)
    decay_rate = hp.learn_decay[1]
    e_steps = hp.learn_decay[2]
    stepsize = floor(hp.epochs / e_steps)
    if hp.epochs - ep_i < stepsize
        return
    elseif (rem(ep_i,stepsize) == 0.0)
        hp.alpha *= decay_rate
        hp.alphaovermb *= decay_rate
        println("     **** at $ep_i stepping down learning rate to $(hp.alpha)")
    else
        return
    end
end


###########################################################################
#  layer functions:  activation and gradients for units of different types
###########################################################################

# two methods for linear layer units, with bias and without
function affine(weights, data, bias)  # with bias
    return weights * data .+ bias
end


function affine(weights, data)  # no bias
    return weights * data
end


function sigmoid!(z::AbstractArray{Float64,2}, a::AbstractArray{Float64,2})
    a[:] = 1.0 ./ (1.0 .+ exp.(-z))
end

function tanh_act!(z::AbstractArray{Float64,2}, a::AbstractArray{Float64,2})
    a[:] = tanh.(z)
end

function l_relu!(z::AbstractArray{Float64,2}, a::AbstractArray{Float64,2}) # leaky relu
    a[:] = map(j -> j >= 0.0 ? j : l_relu_neg * j, z)
end


function relu!(z::AbstractArray{Float64,2}, a::AbstractArray{Float64,2})
    a[:] = max.(z, 0.0)
end


function softmax!(z::AbstractArray{Float64,2}, a::AbstractArray{Float64,2})

    expf = similar(a)
    expf[:] = exp.(z .- maximum(z,1))
    a[:] = expf ./ sum(expf, 1)

end


function regression!(z::AbstractArray{Float64,2}, a::AbstractArray{Float64,2})
    a[:] = z[:]
end


# two methods for gradient of linear layer units:  without bias and with
# not using this yet
function affine_gradient(data, layer)  # no bias
    return data.a[layer-1]'
end


function sigmoid_gradient!(z::AbstractArray{Float64,2}, grad::AbstractArray{Float64,2})
    sigmoid!(z, grad)
    grad[:] = grad .* (1.0 .- grad)
end


function tanh_act_gradient!(z::AbstractArray{Float64,2}, grad::AbstractArray{Float64,2})
    grad[:] = 1.0 .- tanh.(z).^2
end


function l_relu_gradient!(z::AbstractArray{Float64,2}, grad::AbstractArray{Float64,2})
    grad[:] = map(j -> j > 0.0 ? 1.0 : l_relu_neg, z);
end


function relu_gradient!(z::AbstractArray{Float64,2}, grad::AbstractArray{Float64,2})
    grad[:] = map(j -> j > 0.0 ? 1.0 : 0.0, z);
end


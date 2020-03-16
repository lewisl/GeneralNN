# this was an awesome way to pass a list of specific arguments selected from a bigger list
# of arguments while deferring execution.  Love it, but alas found a way that is clearer to 
# write and document.  And easier for "users" to create alternative functions that can plug in:  
# much more obviously.

# sort of slow way to do padding: lots of allocations, still better than broadcasting
function dopad(arr, pad::Int, cdim::Int; padval=0) 
    dims != 4 && error("dims argument value must be 4 for array as 4 dimensional tensor")
    padval = convert(eltype(arr), padval)
    m,n = size(arr)
    c = size(arr,3)
    k = size(arr,4)
    return [(i in 1:pad) || (j in 1:pad) || (i in m+pad+1:m+2*pad) || (j in n+pad+1:n+2*pad) ? padval : 
        arr[i-pad,j-pad, z, cnt] for i=1:m+2*pad, j=1:n+2*pad, z=1:c, cnt=1:k]
end

"""
    macro gen_argset_ff(func, tpl, fname)

This macro allows you to pick a name for the func and create multiple methods for that function name.
Inputs:
    func:  the name of the func that you are passing arguments to
    tpl:   the tuple of arguments to be passed. These must reference the inputs to argset_ff, which are
              inputs available in the feedfwd! loop.
    fname: must be set to GeneralNN.argset

usage example: 
@gen_argset GeneralNN.relu! (dat.a[hl], dat.z[hl]) GeneralNN.argset


Creates the following method:

    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(relu!), dotrain)
        (dat.a[hl], dat.z[hl])
    end

"""
macro gen_argset_ff(func, tpl, fname)  # confirmed that this works: always use GeneralNN.argset as the fname
    return quote
        function $(esc(fname))(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
                               bn::Batch_norm_params, hl::Int, fn::typeof($func), dotrain); 
            $tpl
        end
    end
end


macro gen_argset_back(func, tpl, fname)  # confirmed that this works: always use GeneralNN.argset as the fname
    return quote
        function $(esc(fname))(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
                               bn::Batch_norm_params, hl::Int, fn::typeof($func)); 
            $tpl
        end
    end
end


macro gen_argset_update(func, tpl, fname)  # confirmed that this works: always use GeneralNN.argset as the fname
    return quote
        function $(esc(fname))(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, 
                hl::Int, fn::typeof($func)); 
            $tpl
        end
    end
end


###################################################
# argset methods to pass in the training loop
###################################################
# feed forward feedfwd! passes dotrain argument
    # affine!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(affine!), dotrain)
        (dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
    end
    # affine_nobias!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(affine_nobias!), dotrain)  #TODO: we can take bias out in layer_functions.jl
        (dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
    end
# activation functions
    # sigmoid!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(sigmoid!), dotrain)
        (dat.a[hl], dat.z[hl])
    end
    # tanh_act!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(tanh_act!), dotrain)
        (dat.a[hl], dat.z[hl])
    end
    # l_relu!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(l_relu!), dotrain)
        (dat.a[hl], dat.z[hl])
    end
    # relu!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(relu!), dotrain)
        (dat.a[hl], dat.z[hl])
    end
# classification functions
    # softmax
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(softmax!), dotrain)
        (dat.a[hl], dat.z[hl])
    end
    # logistic!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(logistic!), dotrain)
        (dat.a[hl], dat.z[hl])
    end
    # regression!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(regression!), dotrain)
        (dat.a[hl], dat.z[hl])
    end
    # batch_norm_fwd!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(batch_norm_fwd!), dotrain)
        (dat, bn, hp, hl, dotrain)
    end
    # # batch_norm_fwd_predict!
    # function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
    #     bn::Batch_norm_params, hl::Int, fn::typeof(batch_norm_fwd_predict!))
    #     (dat, bn, hp, hl)
    # end
    # dropout_fwd!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(dropout_fwd!), dotrain)
        (dat.a[hl], hp.droplim[hl], nnw.dropout_mask[hl], dotrain)
    end

    # back propagation backprop! does NOT take dotrain as an input
    # backprop_classify!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(backprop_classify!))
            (dat.epsilon[nnw.output_layer], dat.a[nnw.output_layer], dat.targets)
    end
    # backprop_weights!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(backprop_weights!))
            (nnw.delta_th[hl], nnw.delta_b[hl], dat.epsilon[hl], dat.a[hl-1], hp.mb_size)   
    end
    # backprop_weights_nobias!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(backprop_weights_nobias!))        # TODO fix
            (nnw.delta_th[hl], nnw.delta_b[hl], dat.epsilon[hl], dat.a[hl-1], hp.mb_size)
    end
    # inbound_epsilon!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(inbound_epsilon!))
            (dat.epsilon[hl], nnw.theta[hl+1], dat.epsilon[hl+1])
    end
    # dropout_back!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(dropout_back!))
            (dat.epsilon[hl], nnw.dropout_mask[hl], hp.droplim[hl])    
    end
    # sigmoid_gradient!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(sigmoid_gradient!))
            (dat.grad[hl], dat.z[hl])  
    end
    # tanh_act_gradient!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(tanh_act_gradient!))
            (dat.grad[hl], dat.z[hl])  
    end
    # l_relu_gradient!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(l_relu_gradient!))
            (dat.grad[hl], dat.z[hl])  
    end
    # relu_gradient!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(relu_gradient!))
            (dat.grad[hl], dat.z[hl])  
    end
    # current_lr_epsilon!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(current_lr_epsilon!))
            (dat.epsilon[hl], dat.grad[hl]) 
    end
    # batch_norm_back!
    function argset(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, 
        bn::Batch_norm_params, hl::Int, fn::typeof(batch_norm_back!))   
            (nnw, dat, bn, hl, hp)
    end

    # for update_parameters loop: does NOT take dat or dotrain as iputs
    # update parameters: optimization
    # momentum
    function argset(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, t::Int, fn::typeof(momentum!))   
            (nnw, hp, bn, hl, t)
    end

    # adam
    function argset(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, t::Int, fn::typeof(adam!))   
            (nnw, hp, bn, hl, t)
    end

    # rmsprop
    function argset(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, t::Int, fn::typeof(rmsprop!))   
            (nnw, hp, bn, hl, t)
    end

    # update parameters
    # update_wgts
    function argset(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, t::Int, fn::typeof(update_wgts!))   
        (nnw.theta[hl], nnw.bias[hl], hp.alphamod, nnw.delta_th[hl], nnw.delta_b[hl])
    end

    # update_wgts_nobias
    function argset(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, t::Int, fn::typeof(update_wgts_nobias!))   
        (nnw.theta[hl], hp.alphamod, nnw.delta_th[hl])
    end

    # update_batch_norm
    function argset(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, t::Int, fn::typeof(update_batch_norm!))   
        (bn.gam[hl], bn.bet[hl], hp.alphamod, bn.delta_gam[hl], bn.delta_bet[hl])
    end

    # update_parameters: regularization
    # maxnorm
    function argset(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, 
        t::Int, fn::typeof(maxnorm_reg!))   
            (nnw.theta, hp, hl)
    end

    # l1
    function argset(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, 
        t::Int, fn::typeof(l1_reg!))   
            (nnw.theta, hp, hl)
    end

    # l2
    function argset(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, 
        t::Int, fn::typeof(l2_reg!))   
            (nnw.theta, hp, hl)
    end






    # create model data holding only one example.  Need z, a, and targets only.
    #     this uses an existing example from the dataset
    #     can base a way to do predictions on a small number of samples on this code fragment
    onedat = Batch_view()
    preallocate_minibatch!(onedat::Batch_view, wgts, hp)
    update_batch_views!(onedat, dat, wgts, hp, example:example)


# a macro that creates new argfilt functions
#    but each has a single method because the function names are tripped up by sysgen
macro gen_argfilt(func, tpl)
    return quote
        function argfilt(dat::Union{Model_data, Batch_view}, nnw::Wgts,
             hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, fn::typeof($func)); 
             $tpl
        end
    end
end


# an alternative that is called differently and requires that the resuling functions be used differently
    # this works but is silly clumsy to use
macro gen_argfilt3(func,tpl_f) # tpl_f must be like:  () -> (dat.a[hl], dat.z[hl])
       return quote
           function argfilt(dat::Union{Model_data, Batch_view},nnw::Wgts,
                   hp::Hyper_parameters, bn::Batch_norm_params, hl::Int,fn::typeof($func)) 
                $tpl_f 
            end # function definition
       end # quote block
end # macro definition
# use to generate an argfilt function: af2 = @gen_argfilt2( GeneralNN.relu!, () -> (dat.a[hl],dat.z[hl]) );
# use it the training loop as: relu!(af2(train, nnw, hp, bn, 1,relu!)()...) NOTE empty pair of () to call the
#     anonymous function that was created


# how we used to do the feedfwd function for the training loop
function feedfwd!(dat::Union{Batch_view,Model_data}, nnw, hp, bn, ff_execstack)  
!hp.quiet && println("feedfwd!(dat::Union{Batch_view, Model_data}, nnw, hp)")

    # # dropout for input layer (if probability < 1.0) or noop
    # dropout_fwd_function![1](dat,hp,1)  

    # # hidden layers
    # @fastmath @inbounds for hl = 2:nnw.output_layer-1  
    #     affine_function!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl]) # if do_batch_norm, ignores bias arg
    #     batch_norm_fwd_function!(dat, hl)  # do it or noop
    #     unit_function![hl](dat.a[hl], dat.z[hl]) # per setup_functions
    #     dropout_fwd_function![hl](dat,hp,hl)  # do it or noop
    # end

    # # output layer
    # @inbounds affine!(dat.z[nnw.output_layer], dat.a[nnw.output_layer-1], 
    #                   nnw.theta[nnw.output_layer], nnw.bias[nnw.output_layer])
    # classify_function!(dat.a[nnw.output_layer], dat.z[nnw.output_layer])  # a = activations = predictions

    for lr in 1:hp.n_layers
        for f in ff_execstack[lr]
            f(argfilt(dat, nnw, hp, bn, lr, f)...)
        end
    end

end

# how we used to do the backprop loop for training
"""
function backprop!(nnw, dat, hp)
    Argument nnw.delta_th holds the computed gradients for Wgts, delta_b for bias
    Modifies dat.epsilon, nnw.delta_th, nnw.delta_b in place--caller uses nnw.delta_th, nnw.delta_b
    Use for training iterations
    Send it all of the data or a mini-batch
    Intermediate storage of dat.a, dat.z, dat.epsilon, nnw.delta_th, nnw.delta_b reduces memory allocations
"""
function backprop!(nnw::Wgts, dat::Union{Batch_view,Model_data}, hp, bn, back_execstack)
    !hp.quiet && println("backprop!(nnw, dat, hp)")



    # println("size epsilon of output: ", size(dat.epsilon[nnw.output_layer]))
    # println("size predictions: ", size(dat.a[nnw.output_layer]))
    # println("size targets: ", size(dat.targets))

    # output layer
    @inbounds begin
        # backprop classify
        backprop_classify!(dat.epsilon[nnw.output_layer], dat.a[nnw.output_layer], dat.targets)
            !hp.quiet && println("What is epsilon of output layer? ", mean(dat.epsilon[nnw.output_layer]))
        backprop_weights!(nnw.delta_th[nnw.output_layer], nnw.delta_b[nnw.output_layer],  
            dat.epsilon[nnw.output_layer], dat.a[nnw.output_layer-1], hp.mb_size)   
    end

    # loop over hidden layers
    @fastmath @inbounds for hl = (nnw.output_layer - 1):-1:2  
        # backprop activation
        inbound_epsilon!(dat.epsilon[hl], nnw.theta[hl+1], dat.epsilon[hl+1])
        dropout_back_function![hl](dat, nnw, hp, hl)  # noop if not applicable
        gradient_function![hl](dat.grad[hl], dat.z[hl])  
        current_lr_epsilon!(dat.epsilon[hl], dat.grad[hl]) 

        batch_norm_back_function!(dat, hl)   # noop if not applicable
        backprop_weights_function!(nnw.delta_th[hl], nnw.delta_b[hl], dat.epsilon[hl], dat.a[hl-1], hp.mb_size)

        !hp.quiet && println("what is delta_th $hl? ", nnw.delta_th[hl])
        !hp.quiet && println("what is delta_b $hl? ", nnw.delta_b[hl])
    end
end
    

# here is how we used to do update_parameters!
function update_parameters!(nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, t::Int, update_execstack)  # =Batch_norm_params()
!hp.quiet && println("update_parameters!(nnw, hp, bn)")



    model.optimization_function!(nnw, hp, bn, t)

    # update theta, bias, and batch_norm parameters
    @fastmath @inbounds for hl = 2:nnw.output_layer       
        @inbounds nnw.theta[hl][:] = nnw.theta[hl] .- (hp.alphamod .* nnw.delta_th[hl])
        
        model.reg_function![hl](nnw.theta, hp, hl)  # regularize function per setup.jl setup_functions!

        # @bp

        if hp.do_batch_norm  # update batch normalization parameters
            @inbounds bn.gam[hl][:] .= bn.gam[hl][:] .- (hp.alphamod .* bn.delta_gam[hl])
            @inbounds bn.bet[hl][:] .= bn.bet[hl][:] .- (hp.alphamod .* bn.delta_bet[hl])
        else  # update bias
            @inbounds nnw.bias[hl][:] .= nnw.bias[hl] .- (hp.alphamod .* nnw.delta_b[hl])
        end

    end  

end


function printstruct(st)
    for it in propertynames(st)
        @printf(" %20s %s\n",it, getproperty(st, it))
    end
end

# shows effects of alternative formulas for backprop of batchnorm params
function batch_norm_back!(nnw, dat, bn, hl, hp)
!hp.quiet && println("batch_norm_back!(nnw, dat, bn, hl, hp)")

    mb = hp.mb_size
    @inbounds bn.delta_bet[hl][:] = sum(dat.epsilon[hl], dims=2) ./ mb
    @inbounds bn.delta_gam[hl][:] = sum(dat.epsilon[hl] .* dat.z_norm[hl], dims=2) ./ mb
    @inbounds dat.epsilon[hl][:] = bn.gam[hl] .* dat.epsilon[hl]  # often called delta_z_norm at this stage

    # 1. per Lewis' assessment of multiple sources including Kevin Zakka, Knet.jl
        # good training performance
        # fails grad check for backprop of revised z, but closest of all
        # note we re-use epsilon to reduce pre-allocated memory, hp is the struct of
        # Hyper_parameters, dat is the struct of activation data, 
        # and we reference data and weights by layer [hl],
        # so here is the analytical formula:
        # delta_z = (1.0 / mb) .* (1.0 ./ (stddev .+ ltl_eps) .* (
        #    mb .* delta_z_norm .- sum(delta_z_norm, dims=2) .-
        #    z_norm .* sum(delta_z_norm .* z_norm, dims=2)
        #   )
    @inbounds dat.epsilon[hl][:] = (                               # often called delta_z, dx, dout, or dy
        (1.0 / mb) .* (1.0 ./ (bn.stddev[hl] .+ hp.ltl_eps))  .* (          # added term: .* bn.gam[hl]
            mb .* dat.epsilon[hl] .- sum(dat.epsilon[hl], dims=2) .-
            dat.z_norm[hl] .* sum(dat.epsilon[hl] .* dat.z_norm[hl], dims=2)
            )
        )

    # 2. from Deriving Batch-Norm Backprop Equations, Chris Yeh
        # training slightly worse
        # grad check considerably worse
    # @inbounds dat.delta_z[hl][:] = (                               
    #         (1.0 / mb) .* (bn.gam[hl] ./ bn.stddev[hl]) .*         
    #         (mb .* dat.epsilon[hl] .- (dat.z_norm[hl] .* bn.delta_gam[hl]) .- (bn.delta_bet[hl] * ones(1,mb)))
    #     )

    # 3. from https://cthorey.github.io./backpropagation/
        # worst bad grad check
        # terrible training performance
    # @inbounds dat.z[hl][:] = (
    #     (1.0/mb) .* bn.gam[hl] .* (1.0 ./ bn.stddev[hl]) .*
    #     (mb .* dat.epsilon[hl] .- sum(dat.epsilon[hl],dims=2) .- (dat.z[hl] .- bn.mu[hl]) .* 
    #      (mb ./ bn.stddev[hl] .^ 2) .* sum(dat.epsilon[hl] .* (dat.z_norm[hl] .* bn.stddev[hl] .- bn.mu[hl]),dims=2))
    #     )

    # 4. slow componentized approach from https://github.com/kevinzakka/research-paper-notes/blob/master/batch_norm.py
        # grad check only slightly worse
        # training performance only slightly worse
        # perf noticeably worse, but not fully optimized
    # @inbounds begin # do preliminary derivative components
    #     zmu = similar(dat.z[hl])
    #     zmu[:] = dat.z_norm[hl] .* bn.stddev[hl]
    #     dvar = similar(bn.stddev[hl])
    #     # println(size(bn.stddev[hl]))
    #     dvar[:] = sum(dat.delta_z_norm[hl] .* -1.0 ./ bn.stddev[hl] .* -0.5 .* (1.0./bn.stddev[hl]).^3, dims=2)
    #     dmu = similar(bn.stddev[hl])
    #     dmu[:] = sum(dat.delta_z_norm[hl] .* -1.0 ./ bn.stddev[hl], dims=2) .+ (dvar .* (-2.0/mb) .* sum(zmu,dims=2))
    #     dx1 = similar(dat.delta_z_norm[hl])
    #     dx1[:] = dat.delta_z_norm[hl] .* (1.0 ./ bn.stddev[hl])
    #     dx2 = similar(dat.z[hl])
    #     dx2[:] = dvar .* (2.0 / mb) .* zmu
    #     dx3 = similar(bn.stddev[hl])
    #     dx3[:] = (1.0 / mb) .* dmu
    # end

    # @inbounds dat.delta_z[hl][:] = dx1 .+ dx2 .+ dx3

    # 5. From knet.jl framework
        # exactly matches the results of 1
        # 50% slower (improvement possible)
        # same grad check results, same training results
     # mu, ivar = _get_cache_data(cache, x, eps)
        # x_mu = x .- mu
    # @inbounds begin    
        # zmu = dat.z_norm[hl] .* bn.stddev[hl]
        # # equations from the original paper
        # # dyivar = dy .* ivar
        # istddev = (1.0 ./ bn.stddev[hl])
        # dyivar = dat.epsilon[hl] .* istddev
        # bn.delta_gam[hl][:] = sum(zmu .* dyivar, dims=2) ./ hp.mb_size   # stupid way to do this
        # bn.delta_bet[hl][:] = sum(dat.epsilon[hl], dims=2) ./ hp.mb_size
        # dyivar .*= bn.gam[hl]  # dy * 1/stddev * gam
        # # if g !== nothing
        # #     dg = sum(x_mu .* dyivar, dims=dims)
        # #     db = sum(dy, dims=dims)
        # #     dyivar .*= g
        # # else
        # #     dg, db = nothing, nothing
        # # end
        # # m = prod(d->size(x,d), dims) # size(x, dims...))
        # # dsigma2 = -sum(dyivar .* x_mu .* ivar.^2, dims=dims) ./ 2
        # dsigma2 = -sum(dyivar .* zmu .* istddev.^2, dims = 2) ./ 2.0
        # # dmu = -sum(dyivar, dims=dims) .- 2dsigma2 .* sum(x_mu, dims=dims) ./ m
        # dmu = -sum(dyivar, dims=2) .- 2.0 .* dsigma2 .* sum(zmu, dims=2) ./ mb
        # dat.delta_z[hl][:] = dyivar .+ dsigma2 .* (2.0 .* zmu ./ mb) .+ (dmu ./ mb)
    # end
end
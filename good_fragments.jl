    # create model data holding only one example.  Need z, a, and targets only.
    #     this uses an existing example from the dataset
    #     can base a way to do predictions on a small number of samples on this code fragment
    onedat = Batch_view()
    preallocate_minibatch!(onedat::Batch_view, wgts, hp)
    update_batch_views!(onedat, dat, wgts, hp, example:example)


    
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
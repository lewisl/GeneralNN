

function compute_numgrad!(numgradtheta, numgradbias, wgts, dat, bn, hp; tweak = 1e-7)
    # wgtgrad = zeros(length(wgts))
    # perturb = zeros(length(wgts))

    for lr in 2:wgts.output_layer  # loop by layer
        # perturb[p] = tweak   # peturb a single weight; leave others alone to get partial for a single weight
        for bi in 1:length(wgts.bias[lr])  # tweak each bias, compute partial diff for each bias
            wgts.bias[lr][bi] -= tweak
            feedfwd_predict!(dat, wgts, bn, hp)
            loss1 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n, wgts.theta, hp.lambda, hp.reg,
                                  wgts.output_layer)
            # loss1 = (-1.0 / dat.n) * (dot(dat.targets,log.(dat.a[wgts.output_layer] .+ 1e-50)) +
            #         dot((1.0 .- dat.targets), log.(1.0 .- dat.a[wgts.output_layer] .+ 1e-50)))

            # println("predictions = ", dat.a[wgts.output_layer])
            # println("targets =     ", dat.targets)
            # println("loss1 =       ", loss1)
            # println("n =           ", dat.n)

            wgts.bias[lr][bi] += 2.0 * tweak
            feedfwd_predict!(dat, wgts, bn, hp)
            loss2 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n, wgts.theta, hp.lambda, hp.reg,
                                  wgts.output_layer)  
            # println("loss2 =       ", loss2)  
            wgts.bias[lr][bi] -= tweak      
            numgradbias[lr][bi] = (loss2 - loss1) / (2.0 * tweak)
        end
        for thi in eachindex(wgts.theta[lr])  # tweak each theta, compute partial diff for each theta
            wgts.theta[lr][thi] -= tweak
            feedfwd_predict!(dat, wgts, bn, hp)
            loss1 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n, wgts.theta, hp.lambda, hp.reg,
                                  wgts.output_layer)
            wgts.theta[lr][thi] += 2.0 * tweak
            feedfwd_predict!(dat, wgts, bn, hp)
            loss2 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n, wgts.theta, hp.lambda, hp.reg,
                                  wgts.output_layer)
            wgts.theta[lr][thi] -= tweak      
            numgradtheta[lr][thi] = (loss2 - loss1) / (2.0 * tweak)
        end
    end

end

function compute_modelgrad!(dat, nnw, bn, hp)
    # no optimization, no batch normalization, no minibatches, no learning rate (e.g. alpha = 1)
    # can do with or without regularization if L2--has to be consistent for numgrad and modelgrad
    #    but there is no reason to do it

    feedfwd!(dat, nnw, bn, hp)  # for all layers
    backprop!(nnw, bn, dat, hp)  # for all layers   

    # now we just need 1/n .* nnw.delta_w   and 1/mb .* nnw.delta_b
    nnw.delta_b .*= 1.0/dat.n
    nnw.delta_w .*= 1.0/dat.n
end

function check_grads(hp)
    println("  ****** Testing gradient calculation on minitiature dataset")
    # set mini-model size
    in_k = 3  # input_layer_size = features
    hidden_layer_size = 5
    out_k = 3   # num_labels
    m = 5

    # set appropriate bogus hyper-parameters for mini-model
    minihp = Hyper_parameters()
        minihp.reg = ""                # no reg
        minihp.plots = []              # no plots
        minihp.plot_now = false        # no plots
        minihp.opt = "none"            # no optimization
        minihp.do_learn_decay = false  # no optimization
        minihp.do_batch_norm = false   # no batches
        minihp.dobatch = false         # no batches
        minihp.alpha = 1.0             # no learning rate

        # use the real models params here
        minihp.hidden = hp.hidden

    minibn = Batch_norm_params()  # don't use; don't need real values; must supply input argument to use training code

    # set hidden layer unit count 
    for hl in 1:length(minihp.hidden)
        minihp.hidden[hl] =  (minihp.hidden[hl][1], hidden_layer_size)  # can't update the tuple; replace it
    end

    # initialize weights and bias for mini-model
    miniwgts = Wgts()

    # set layer sizes
    miniwgts.output_layer = 2 + size(minihp.hidden, 1) # input layer is 1, output layer is highest value
    miniwgts.ks = [in_k, map(x -> x[2], minihp.hidden)..., out_k]       # no. of output units by layer

    # set dimensions of the linear Wgts for each layer and initialize
    push!(miniwgts.theta_dims, (in_k, 1)) # weight dimensions for the input layer -- if using array, must splat as arg
    for l = 2:miniwgts.output_layer  
        push!(miniwgts.theta_dims, (miniwgts.ks[l], miniwgts.ks[l-1]))
    end

    miniwgts.theta = [zeros(2,2)] # layer 1 not used
    for l = 2:miniwgts.output_layer
        push!(miniwgts.theta, debug_initialize_weights!(zeros(miniwgts.theta_dims[l]))) # sqrt of no. of input units
    end

    # set dimensions of bias terms and initialize
    miniwgts.bias = bias_zeros(miniwgts.ks)
    for l = 1:length(miniwgts.bias)
        debug_initialize_weights!(miniwgts.bias[l])
    end

    # initialize model gradients
    miniwgts.delta_w = deepcopy(miniwgts.theta)
    miniwgts.delta_b = deepcopy(miniwgts.bias)

    # initialize numeric gradients
    numgradtheta = deepcopy(miniwgts.theta)
    numgradbias  = deepcopy(miniwgts.bias)

    # create repeatable bogus train data (x,y) for mini-model
    minidat = Model_data()
    minidat.inputs = zeros(in_k, m)
    minidat.targets = zeros(out_k, m)
    minidat.n = m; minidat.in_k = in_k; minidat.out_k = out_k

    debug_initialize_weights!(minidat.inputs)

    minidat.targets = 1 .+ mod.(collect(1:m), out_k)
    minidat.targets = collect(onehot(minidat.targets, out_k)')  # onehot encode categories

    # preallocate the training matrices
    preallocate_data!(minidat, miniwgts, m, minihp)

    # advance 5 iterations using the analytic model

    # compute numgrad
    println("  ****** Calculating numeric approximation gradients")
    compute_numgrad!(numgradtheta, numgradbias, miniwgts, minidat, minibn, minihp; tweak=1e-6)
    numgrad = (numgradtheta,numgradbias)

    # compute_modelgrad!()
    println("  ****** Calculating feedfwd/backprop gradients")
    compute_modelgrad!(minidat, miniwgts, minibn, minihp)
    modgrad = (deepcopy(miniwgts.delta_w), deepcopy(miniwgts.delta_b))

    deltacols = hcat(flat(modgrad), flat(numgrad))
    relative_error = map(x -> abs(x[1]-x[2]) / (maximum(abs.(x)) + 1e-15), eachrow(deltacols))
    println("\nrelative errors")
    for v in relative_error;
        @printf("\n  % 8.5f", v) 
    end


    #  compare modelgrad and numgrad
    println("\ngradients")
    println("    model", "      numeric")
    printby2(deltacols)


end

"""
**debug_initialize_weights** Initialize the weights of a layer 
using a fixed strategy, which will help you later in debugging.

We use this with a modeling approach that separates weights from the bias term.
"""
function debug_initialize_weights!(mx)

    # Initialize w using "sin", this ensures that w is always of the same
    # values and will be useful for debugging

    mx[:] = reshape(sin.(0:length(mx)-1), size(mx)) / 10.0

end

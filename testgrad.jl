# hp = setup_training("nninputs.toml")
# train_x, train_y = extract_data("digits5000by400.mat")
# train, mb, nnw, bn = pretrain(train_x, train_y, hp)
# statsdat = setup_stats(hp, dotest)  
# training_loop to run some epochs--start with minibatch, optimization, etc.
#     then check grads in one epoch without that stuff


# method to use fully built model
function check_grads(hp, train_x, train_y, iters; samplepct=.005, quiet=false, tweak=1e-6)
    println("  ****** Testing gradient calculation on input training dataset")

    println("  ****** Setting up model and data")
    # advance the training loop with model hyper-parameters
    dotest = false
    train, mb, nnw, bn = pretrain(train_x, train_y, hp)
    statsdat = setup_stats(hp, dotest)  

    # set iters to train before grad testing
    @printf("  ****** Advancing model %d iterations\n", iters)
        olditers = hp.epochs
        oldalpha = hp.alpha

        hp.epochs = iters
        training_time = training_loop!(hp, train, mb, nnw, bn, statsdat)

        hp.epochs = olditers
        hp.alpha = oldalpha

    # set new hyper-parameters for gradient checking
    minihp = set_hp(hp)
    setup_functions!(minihp, nnw, bn, train)  

    run_check_grads(minihp, nnw, train; samplepct=samplepct, tweak=tweak, quiet=quiet)
end


# method to use mini-model with same number/type of hidden layers
function check_grads(hp; in_k=3, out_k=3, m=5, hid=5, iters=100, tweak=1e-6, quiet=false)
    println("  ****** Testing gradient calculation on minitiature dataset")

    println("  ****** Setting up model and test data")
    # set new hyper-parameters for gradient checking
    minihp = set_hp(hp)

    # create minidata and mini-model
    minidat, minimb, miniwgts, minibn, ministats = use_minidata(minihp;
        in_k=in_k, out_k=out_k, m=m, hid=hid)   

    # advance many iterations using the analytic model
    @printf("  ****** Advancing model %d iterations", iters)
    minihp.epochs = iters
    training_time = training_loop!(minihp, minidat, minimb, miniwgts, minibn, ministats)
end


function set_hp(hp)
    # set appropriate bogus hyper-parameters for mini-model
    minihp = Hyper_parameters()
        minihp.reg = ""                # no reg
        minihp.stats = []              # no stats
        minihp.plot_now = false        # no plots
        minihp.opt = "none"            # no optimization
        minihp.do_learn_decay = false  # no optimization
        minihp.do_batch_norm = false   # no batches
        minihp.dobatch = false         # no batches
        minihp.alpha = 1.0 #hp.alpha        # no learning rate
        minihp.classify = hp.classify
        minihp.dropout = false
        minihp.norm_mode = "standard"  # only used with minimodel
        minihp.bias_initializer = hp.bias_initializer  # only used with minimodel

        # use the real models params here
        minihp.hidden = hp.hidden
    return minihp
end


function run_check_grads(hp, wgts, dat; samplepct=.015, tweak=1e-6, quiet=false)

    # sample indices of grads
    biasgradidx = [sample_idx(samplepct,x) for x in wgts.bias[2:wgts.output_layer]]
    insert!(biasgradidx, 1, [])  # placeholder for layer 1
    thetagradidx = [sample_idx(samplepct,x) for x in wgts.theta[2:wgts.output_layer]]
    insert!(thetagradidx, 1, [])  # placeholder for layer 1

    # initialize numeric gradients results holder
    numgradtheta = [zeros(length(x)) for x in thetagradidx]  # deepcopy(wgts.theta)      # deepcopy(miniwgts.theta)
    numgradbias  = [zeros(length(x)) for x in biasgradidx]  # deepcopy(wgts.bias)       # deepcopy(miniwgts.bias)

    # capture starting data structures
    startwgts = deepcopy(wgts)
    startdat = deepcopy(dat)  # do we need this?

    # compute numgrad
    compute_numgrad!(numgradtheta, numgradbias, thetagradidx, biasgradidx, wgts, dat, hp; tweak=tweak)
    numgrad = (numgradtheta,numgradbias)

    #recover starting data structures
    wgts = deepcopy(startwgts)
    dat = deepcopy(startdat)

    # compute_modelgrad!()
    println("  ****** Calculating feedfwd/backprop gradients")
    compute_modelgrad!(dat, wgts, hp)
    modgrad = (deepcopy(wgts.delta_w), deepcopy(wgts.delta_b))

    # filter modgrads to match sample of numgrads
    modgrad = [   [modgrad[1][i][thetagradidx[i]] for i in 2:wgts.output_layer],     
                  [modgrad[2][i][biasgradidx[i]] for i in 2:wgts.output_layer]     ]

    deltacols = hcat(flat(modgrad), flat(numgrad))
    relative_error = map(x -> abs(x[1]-x[2]) / (maximum(abs.(x)) + 1e-15), eachrow(deltacols))
   
    !quiet && begin
        println("\nrelative errors")
        for v in relative_error;
            @printf("  % 8.5f\n", v) 
        end

        #  compare modelgrad and numgrad
        println("\ngradients")
        println("    model", "      numeric")
        # print theta gradients
        startrow = 1
        endrow = 0
        for l = 2:wgts.output_layer
            @printf("theta of layer %d, sample size %s, actual size %s\n", l, size(numgradtheta[l]), size(wgts.theta[l]))
            endrow = startrow + length(numgradtheta[l]) - 1
            @printf("start: %d   end: %d\n", startrow, endrow)

            printby2(deltacols[startrow:endrow,:])
            startrow = endrow + 1
        end

        # print bias gradients
        startrow = endrow + 1
        for l = 2:wgts.output_layer
            @printf("bias of layer %d, sample size %s, actual size %s\n", l, size(numgradbias[l]), size(wgts.bias[l]))
            endrow = startrow + length(numgradbias[l]) - 1
            @printf("start: %d   end: %d\n", startrow, endrow)
            printby2(deltacols[startrow:endrow,:])
            startrow = endrow + 1
        end 
    end

    # summary stats
    println("\nMean Gross difference")
    println(mean(abs.(deltacols[:,1] .- deltacols[:,2])))
    println("\nMean Relative error")
    println(mean(relative_error))
    println("\nMax Relative error")
    println(maximum(relative_error))
    println("\nCount Relative error >= 0.2")
    println(count(i -> i >= 0.2, relative_error), " out of ", length(relative_error))
end


function use_minidata(minihp;in_k=3, out_k=3, m=5, hid=5)

    # change hidden layer unit count to tiny example
    for hl in 1:length(minihp.hidden)
        minihp.hidden[hl] =  (minihp.hidden[hl][1], hid)  # can't update the tuple; replace it
    end

    # create repeatable, bogus train data (x,y) for mini-model
    mini_x = zeros(in_k, m)
    mini_x = debug_initialize_weights!(mini_x)
    mini_y = 1 .+ mod.(collect(1:m), out_k)
    mini_y = collect(onehot(mini_y, out_k)')  # onehot encode categories

    # normalize_inputs!(mini_x, "standard")

    # create training data structures
    minidat, minimb, miniwgts, minibn = pretrain(mini_x, mini_y, minihp)

    # repeatable initization of theta and bias
    for l = 2:miniwgts.output_layer
        # debug_initialize_weights!(miniwgts.theta[l])
        xavier_initialize_weights!(miniwgts.theta[l])
        bias_val!(miniwgts.bias[l], minihp.bias_initializer)
    end

    dotest = false
    ministats = setup_stats(minihp, dotest)  

    return minidat, minimb, miniwgts, minibn, ministats
end


function compute_numgrad!(numgradtheta, numgradbias, thetagradidx, biasgradidx, wgts, dat, hp; tweak = 1e-7)

    println("  ****** Calculating numeric bias gradients")
    for lr in 2:wgts.output_layer  # loop by layer
        for (idx,bi) in enumerate(biasgradidx[lr])   # eachindex(wgts.bias[lr])  # tweak each bias, compute partial diff for each bias
            wgts.bias[lr][bi] -= tweak
            feedfwd_predict!(dat, wgts, hp)       #feedfwd_predict!(dat, wgts, hp)
            loss1 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n, wgts.theta, hp.lambda, hp.reg,
                                  wgts.output_layer)

            wgts.bias[lr][bi] += 2.0 * tweak
            feedfwd_predict!(dat, wgts, hp)
            loss2 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n, wgts.theta, hp.lambda, hp.reg,
                                  wgts.output_layer)  

            wgts.bias[lr][bi] -= tweak   

            numgradbias[lr][idx] = (loss2 - loss1) / (2.0 * tweak)
        end
    end

    println("  ****** Calculating numeric theta gradients")
    for lr in 2:wgts.output_layer  # loop by layer
        for (idx,thi) in enumerate(thetagradidx[lr])  # eachindex(wgts.theta[lr])  # tweak each theta, compute partial diff for each theta

            wgts.theta[lr][thi] -= tweak
            feedfwd_predict!(dat, wgts, hp)
            loss1 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n, wgts.theta, hp.lambda, hp.reg,
                                  wgts.output_layer)

            wgts.theta[lr][thi] += 2.0 * tweak
            feedfwd_predict!(dat, wgts, hp)
            loss2 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n, wgts.theta, hp.lambda, hp.reg,
                                  wgts.output_layer)

            wgts.theta[lr][thi] -= tweak   

            numgradtheta[lr][idx] = (loss2 - loss1) / (2.0 * tweak)
        end
    end
end  # function compute_numgrad!


function compute_modelgrad!(dat, nnw, hp)
    # no optimization, no reg, no batch normalization, no minibatches, no learning rate (e.g. alpha = 1)

    feedfwd!(dat, nnw, hp)  # for all layers
    backprop!(nnw, dat, hp)  # for all layers   

    # now we just need 1/n .* nnw.delta_w   and 1/mb .* nnw.delta_b
    nnw.delta_b ./= dat.n
    nnw.delta_w ./= dat.n
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

function xavier_initialize_weights!(mx, scale=2.0)
    mx[:] = randn(size(mx)) .* sqrt(scale/size(mx,2))
end

function bias_val!(bias, val)
    fill!(bias, val) 
end

function bias_rand!(bias, val)
    bias[:] = rand(size(bias,1))
end


function sample_idx(xpct, x)
    lx = length(x)
    xcnt = ceil(Int, xpct * lx) + 2
    indices = rand(1:lx, xcnt)
end
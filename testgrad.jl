# hp = setup_training("nninputs.toml")
# train_x, train_y = extract_data("digits5000by400.mat")
# shuffle_data!(train_x, train_y)
# train, mb, nnw, bn = pretrain(train_x, train_y, hp)
# statsdat = setup_stats(hp, dotest)  
# training_loop to run some epochs--start with minibatch, optimization, etc.
#     then check grads in one epoch without that stuff


# method for using the full model and training data to check grads
function prep_check(hp, train_x, train_y, iters; samplepct=.005, quiet=false, tweak=1e-6)
    # advance the training loop with model hyper-parameters
    dotest = false
    train, mb, nnw, bn = pretrain(train_x, train_y, hp)
    statsdat = setup_stats(hp, dotest)  

    # set iters to train before grad testing
    @printf("  ****** Advancing model %d iterations\n", iters)
        olditers = hp.epochs
        oldalpha = hp.alpha
        oldstats = hp.stats

        hp.epochs = iters
        hp.stats = []
        training_time = training_loop!(hp, train, mb, nnw, bn, statsdat)
        output_stats(train,nnw, bn, hp,1.1, statsdat)

        hp.epochs = olditers
        hp.alpha = oldalpha
        hp.stats = oldstats

    # set new hyper-parameters for gradient checking
    # minihp = set_hp(hp)
    minihp = hp
    setup_functions!(minihp, nnw, bn, train)  

    return minihp, nnw, train # TODO do we also need to return  mb, bn, statsdat?
end


# TODO test the full model version again
# method to use fully built model with real training data
function check_grads(hp, train_x, train_y, iters; samplepct=.005, quiet=false, tweak=1e-6, post_iters=0,
    biasidx = [], thetaidx = [])
    println("  ****** Testing gradient calculation on input training dataset")

    println("  ****** Setting up model and data")

    minihp, nnw, train = prep_check(hp, train_x, train_y, iters; samplepct=samplepct, quiet=quiet, tweak=tweak)

    run_check_grads(minihp, nnw, train; samplepct=samplepct, tweak=tweak, quiet=quiet, iters=post_iters,
                    biasgradidx = biasidx, thetagradidx = thetaidx)
end


# method to use mini-model with same number/type of hidden layers as designed model
function prep_check(hp; in_k=3, out_k=3, m=5, hid=5, iters=100, samplepct=1.0, tweak=1e-4, quiet=false)

    # set new hyper-parameters for gradient checking
    minihp = set_hp(hp)

    # create minidata and mini-model
    @bp
    minidat, minimb, miniwgts, minibn, ministats = use_minidata(minihp;
        in_k=in_k, out_k=out_k, m=m, hid=hid)   

    # advance some iterations using the analytic model
    @printf("\n  ****** Advancing model %d iterations\n", iters)
    minihp.epochs = iters

    training_time = training_loop!(minihp, minidat, minimb, miniwgts, minibn, ministats)

    return minihp, miniwgts, minidat
end


# method to use mini-model with same number/type of hidden layers as designed model
function check_grads(hp; in_k=3, out_k=3, m=5, hid=5, iters=30, samplepct=1.0, tweak=1e-4, quiet=false,
    post_iters=0, biasidx = [], thetaidx = [])
    println("  ****** Testing gradient calculation on minitiature dataset")

    println("  ****** Setting up model and test data")

    @bp
    # run prep_check
    minihp, miniwgts, minidat = prep_check(hp; in_k=in_k, out_k=out_k, m=m, hid=hid, iters=iters,
        samplepct=samplepct, tweak=tweak, quiet=false)

    @bp
    run_check_grads(minihp, miniwgts, minidat; samplepct=samplepct, tweak=tweak, quiet=quiet)
end


function compare_tweak(hp, wgts, dat; eps_wgt=(2,3,"bias"), example=10)
    tweaks = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    @printf("%16s %16s %16s %16s %16s %16s \n", "cost",   "costnew",  "diff",  "tweak*w", "err",     "tweak")

    for tweak in tweaks
        cost, costnew, w = check_one(hp, wgts, dat; tweak=tweak, eps_wgt=eps_wgt, example=example)
        @printf("%16.12f  %16.12f %16.12f   %16.12f   %16.12f  %16.12f\n", cost, costnew, costnew-cost, tweak*w, 
                                                        (costnew-cost)-tweak*w, tweak)
    end
end

# eps_wgt = (selr, idx, typ) where lr is int of layer (either hidden or output) 
#                                idx is the linear index of theta or bias for lyr
#                                typ must be "theta" or "bias"
function check_one(hp, inwgts, indat; tweak=1e-6, eps_wgt=(2,3,"bias"), example=10)  
    wgts = deepcopy(inwgts)
    dat = deepcopy(indat)
    selr = eps_wgt[1]
    idx = eps_wgt[2]
    typ = eps_wgt[3]
    n_layers = wgts.output_layer
    deltype = if typ == "theta" 
                    "delta_w" 
              elseif typ == "bias" 
                  "delta_b" 
              else
                  error("Third element of eps_wgt must be either theta or bias, was $typ")
              end

    # create model data holding only one example.  Need z, a, and targets only.
    onedat = Batch_view()  # holder
    preallocate_minibatch!(onedat::Batch_view, wgts, hp)  # create array layers that match the model
    update_batch_views!(onedat, dat, wgts, hp, example:example)  # slice out column example

    # 1. get the cost for a single example at current weights
    feedfwd_predict!(onedat, wgts, hp) 
    cost = cost_function(onedat.targets[:,1], onedat.a[wgts.output_layer][:,1], 1, wgts.theta, hp.lambda, hp.reg,
                          wgts.output_layer)
    
    # 2. compute the gradient of the weight we will perturb by tweak (not yet...)
    backprop!(wgts, dat, hp)  # for all layers   
    gradone = getproperty(wgts, Symbol(deltype))[selr][idx]  # delta_w or delta_b

    # 3. calculate the loss for the perturbed weight
    w = getproperty(wgts, Symbol(typ))[selr][idx]  # bias or theta
    w_plus = w + tweak
    getproperty(wgts, Symbol(typ))[selr][idx] = w_plus  # set the chosen weight to tweaked value
    feedfwd_predict!(onedat, wgts, hp) 
    # cost_plus = cost_function(onedat.targets[:,1], onedat.a[wgts.output_layer][:,1], 1, wgts.theta, hp.lambda, hp.reg,
                          # wgts.output_layer)
    costnew = cost_function(onedat.targets[:,1], onedat.a[wgts.output_layer][:,1], 1, wgts.theta, hp.lambda, hp.reg,
                          wgts.output_layer)

    # w_minus = w - tweak
    # getproperty(wgts, Symbol(typ))[selr][idx] = w_minus  # set the chosen weight to tweaked value
    # feedfwd_predict!(onedat, wgts, hp) 
    # cost_minus = cost_function(onedat.targets[:,1], onedat.a[wgts.output_layer][:,1], 1, wgts.theta, hp.lambda, hp.reg,
    #                       wgts.output_layer)    
    # costnew = (cost_plus + cost_minus) / (2.0)   # centered cost difference, can also use single-sided

    # 4. compare to test value: asserts that difference in the costs is the gradient scaled by the tweak,
    #        which is a crude way to show the diff as the limit of tweak approaches zero
    costcheck = cost + (tweak * gradone)

    # println("Original Cost: ", cost, " tweaked cost: ", costnew)
    # println("Original cost + delta:                          ",costcheck)
    # println("Diff tweaked cost - (orig. cost + delta) ", costnew - costcheck)
    return cost, costnew, gradone
end


function run_check_grads(hp, wgts, dat; samplepct=.015, tweak=1e-6, quiet=false, iters=0,
    biasgradidx = [], thetagradidx = [])

    if !(samplepct > 0.0 && samplepct <= 1.0)
        error("input samplepct must be greater than zero and less than or equal to 1.0")
    end

    if length(biasgradidx) == 0  # no input of pre-selected indices
        if samplepct == 1.0
            biasgradidx = [eachindex(x) for x in wgts.bias[2:wgts.output_layer]]
        else  # must be in (0.0,1.0)
            biasgradidx = [sample_idx(samplepct,x) for x in wgts.bias[2:wgts.output_layer]]
        end
        insert!(biasgradidx, 1, eachindex([]))  # placeholder for layer 1
    end

    if length(thetagradidx) == 0  # no input of pre-selected indices
        if samplepct == 1.0
            thetagradidx = [eachindex(x) for x in wgts.theta[2:wgts.output_layer]]
        else  # must be in (0.0,1.0)
            thetagradidx = [sample_idx(samplepct,x) for x in wgts.theta[2:wgts.output_layer]]
        end
        insert!(thetagradidx, 1, eachindex([]))  # placeholder for layer 1
    end
    
    # run model a few more iterations:  especially valuable if full model uses minibatches
    for i = 1:iters
        feedfwd_predict!(dat, wgts, hp) # no batches, no batchnorm, no dropout
        backprop!(wgts, dat, hp)
        update_parameters!(wgts, hp)
    end

    # initialize numeric gradients results holder
    numgradtheta = [zeros(length(x)) for x in thetagradidx]  # deepcopy(wgts.theta)      # deepcopy(miniwgts.theta)
    numgradbias  = [zeros(length(x)) for x in biasgradidx]  # deepcopy(wgts.bias)       # deepcopy(miniwgts.bias)

    # capture starting data structures
    startwgts = deepcopy(wgts)
    startdat = deepcopy(dat)  # do we need this?

    # compute numgrad
    kinkrejecttheta, kinkrejectbias, numcost,  numpreds = compute_numgrad!(numgradtheta, numgradbias, 
                        thetagradidx, biasgradidx, wgts, dat, hp; tweak=tweak)
    numgrad = (numgradtheta,numgradbias)

    #recover starting data structures
    wgts = deepcopy(startwgts)
    dat = deepcopy(startdat)

    # compute_modelgrad!()
    println("  ****** Calculating feedfwd/backprop gradients")
    @bp
    modcost, modpreds = compute_modelgrad!(dat, wgts, hp)  # changes dat and wgts
    modgrad = (deepcopy(wgts.delta_w), deepcopy(wgts.delta_b))  # do I need to copy since compute_numgrad! does no backprop?

    
    # compare results from modgrads and numgrads
     println("  ****** Are costs the same? (model, num cost) ")
     println(modcost, " ", numcost)
     println("  ****** How close are the predictions?")
     println(count(modpreds .== numpreds), " of ", length(modpreds))
     println("      mean abs difference ",sum(abs.(modpreds .- numpreds)) / length(modpreds))

    # filter modgrads to match sample of numgrads
    modgrad = [   [modgrad[1][i][thetagradidx[i]] for i in 2:wgts.output_layer],     
                  [modgrad[2][i][biasgradidx[i]] for i in 2:wgts.output_layer]     ]

    # filter gradients where kink encountered
    # println(kinkrejecttheta)
    # println(kinkrejectbias)

    deltacols = hcat(flat(modgrad), flat(numgrad))
    relative_error = map(x -> abs(x[1]-x[2]) / (maximum(abs.(x)) + 1e-15), eachrow(deltacols))
    rel_err2 = norm(deltacols[1] .- deltacols[2]) / (norm(deltacols[2]) + norm(deltacols[1]))
   
    !quiet && begin
        println("\nrelative errors")
        for v in relative_error;
            @printf("  % 8.5f\n", v) 
        end
        println(rel_err2)

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
    println("\nRelative difference using norm")
    println(norm(deltacols[:,1] .- deltacols[:,2])/norm(deltacols[:,1] .+ deltacols[:,2]))
    println("\nMean Relative error")
    println(mean(relative_error))
    println("\nMax Relative error")
    println(maximum(relative_error))
    println("\nCount Relative error >= 0.2")
    println(count(i -> i >= 0.2, relative_error), " out of ", length(relative_error))
end


function set_hp(hp)
    # set appropriate bogus hyper-parameters for mini-model
    minihp = Hyper_parameters()

        # change parameters to special values for grad checking
        minihp.reg = ""                # no reg
        minihp.stats = []              # no stats
        minihp.plot_now = false        # no plots
        minihp.opt = "none"            # no optimization
        minihp.learn_decay = [1.0, 1.0]  # no optimization
        minihp.do_batch_norm = false   # no batches
        minihp.dobatch = false         # no batches
        minihp.alpha = 0.2             # low learning rate
        minihp.dropout = false
        minihp.norm_mode = "minmax"  # only used with minimodel
        minihp.bias_initializer = hp.bias_initializer  # only used with minimodel
        minihp.lambda = 0.0          # shouldn't matter as reg is ""
        minihp.sparse = false
        minihp.quiet = true

        # use the real models params here
        minihp.hidden = hp.hidden
        minihp.classify = hp.classify

        printstruct(minihp)
        printstruct(hp)

    return minihp
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

    # create training data structures and setup functions, etc (see pretrain function in setup_training.jl)
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

"""
caution 
Kinks in the objective. One source of inaccuracy to be aware of during gradient checking is the problem of kinks. Kinks refer to non-differentiable parts of an objective function, introduced by functions such as ReLU (max(0,x)), or the SVM loss, Maxout neurons, etc. Consider gradient checking the ReLU function at x=−1e6. Since x<0, the analytic gradient at this point is exactly zero. However, the numerical gradient would suddenly compute a non-zero gradient because f(x+h) might cross over the kink (e.g. if h>1e−6) and introduce a non-zero contribution. You might think that this is a pathological case, but in fact this case can be very common. For example, an SVM for CIFAR-10 contains up to 450,000 max(0,x) terms because there are 50,000 examples and each example yields 9 terms to the objective. Moreover, a Neural Network with an SVM classifier will contain many more kinks due to ReLUs.

Note that it is possible to know if a kink was crossed in the evaluation of the loss. This can be done by keeping track of the identities of all “winners” in a function of form max(x,y); That is, was x or y higher during the forward pass. If the identity of at least one winner changes when evaluating f(x+h) and then f(x−h), then a kink was crossed and the numerical gradient will not be exact.
"""


function compute_numgrad!(numgradtheta, numgradbias, thetagradidx, biasgradidx, wgts, dat, hp; tweak = 1e-7)

    @bp

    kinkcnt = 0
    kinkrejectbias = Tuple{Int64, Int64}[]
    kinkrejecttheta = Tuple{Int64, Int64}[]
    comploss = 0.0
    numpreds = deepcopy(dat.a[wgts.output_layer])

    olddat = deepcopy(dat)
    output_layer = wgts.output_layer

    for lr in 2:wgts.output_layer  # loop by layer
        @printf("  ****** Calculating numeric bias gradients layer: %d\n", lr)

        for (idx,bi) in enumerate(biasgradidx[lr])   # eachindex(wgts.bias[lr])  # tweak each bias, compute partial diff for each bias

            wgts.bias[lr][bi] += tweak

            feedfwd_predict!(dat, wgts, hp)       #feedfwd_predict!(dat, wgts, hp)
            kinktst1 = findkink(olddat, dat, output_layer)
            loss1 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) 

            wgts.bias[lr][bi] -= 2.0 * tweak

            feedfwd_predict!(dat, wgts, hp)
            kinktst2 = findkink(olddat, dat, output_layer)
            loss2 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n)   

            wgts.bias[lr][bi] += tweak   

            if kinktst1  || kinktst2 
                kinkcnt += 1
                push!(kinkrejectbias, (lr, bi))
            end

            numgradbias[lr][idx] = (loss1 - loss2) / (2.0 * tweak)  

            # if idx == 1
                @printf("loss1: %f loss2: %f grad: %f kink: %d\n", loss1, loss2, numgradbias[lr][idx], kinkcnt)
            # end
        end

        println("kink errors for bias = ", kinkcnt)

        @printf("  ****** Calculating numeric theta gradients layer: %d\n", lr)

        for (idx,thi) in enumerate(thetagradidx[lr])  # eachindex(wgts.theta[lr])  # tweak each theta, compute partial diff for each theta

            # println("theta: ","layer: ", lr, " index: ", idx)

            wgts.theta[lr][thi] += tweak

            feedfwd_predict!(dat, wgts, hp)
            kinktst1 = findkink(olddat, dat, output_layer)
            loss1 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) 

            wgts.theta[lr][thi] -= 2.0 * tweak

            feedfwd_predict!(dat, wgts, hp)
            kinktst2 = findkink(olddat, dat, output_layer)
            loss2 = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n)

            wgts.theta[lr][thi] += tweak   
            comploss = cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) 

            if kinktst1 || kinktst2 
                kinkcnt += 1
                push!(kinkrejecttheta, (lr, thi))
                # println("Compare tweaked weights for theta kink +: ",wgts.theta[lr][thi]+tweak, " -: ",wgts.theta[lr][thi]-tweak)
            end

            # if idx % 20 == 0
                @printf("loss1: %f loss2: %f grad: %f kink: %d\n", loss1, loss2, numgradtheta[lr][idx], kinkcnt)

            # end

            numgradtheta[lr][idx] = (loss1 - loss2) / (2.0 * tweak)

            # if idx % 20 == 0
                @printf("loss1: %f loss2: %f grad: %f kink: %d\n", loss1, loss2, numgradtheta[lr][idx], kinkcnt)

            # end

            numpreds = dat.a[wgts.output_layer]
        end
    end
    println("count of gradients rejected: ", kinkcnt)
    return kinkrejecttheta, kinkrejectbias, comploss, numpreds
end  # function compute_numgrad!


function findkink(olddat, dat, output_layer)
    kink = false
    for lr = 2:output_layer
        kink = !all(sign.(olddat.a[lr]) .== sign.(dat.a[lr]))
        if kink 
            break
        end
    end
    return kink
end


function compute_modelgrad!(dat, nnw, hp)
    # no optimization, no reg, no batch normalization, no minibatches, no learning rate (e.g. alpha = 1)

    @bp

    feedfwd_predict!(dat, nnw, hp)  # for all layers

    # println("first pass of modelgrad calc")
    # println(dat.a)

    basecost = cost_function(dat.targets, dat.a[nnw.output_layer], dat.n)

    # println("cost of first prediction in modelgrad")
    # println(basecost)

    @bp

    backprop!(nnw, dat, hp)  # for all layers   

    # now we just need 1/n .* nnw.delta_w   and 1/n .* nnw.delta_b
    nnw.delta_b[:] ./= dat.n
    nnw.delta_w[:] ./= dat.n

    return basecost, dat.a[nnw.output_layer]
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


# method works on the number of elements you want sampled indices for 
function sample_idx(xpct, lx::Int)
    xcnt = ceil(Int, xpct * lx) + 2
    sort!(randperm(lx)[1:xcnt])
end


# method works on the object that you want indices for
function sample_idx(xpct, x::Array{T}) where {T}
    lx = length(x)
    sample_idx(xpct, lx)
end


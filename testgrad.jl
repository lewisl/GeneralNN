# hp = setup_params("nninputs.toml")
# train_x, train_y = extract_data("digits5000by400.mat")
# shuffle_data!(train_x, train_y)
# train, mb, nnw, bn = pretrain(train_x, train_y, hp)
# stats = setup_stats(hp, dotest)  
# training_loop to run some epochs--start with minibatch, optimization, etc.
#     then check grads in one epoch without that stuff


# method for using the full model and training data to check grads
function prep_check(hp, train_x, train_y, iters; samplepct=.005, quiet=false, tweak=1e-6)

    # prepare the model (sub-steps that are part of train() )
    dotest = false
    train, mb, nnw, bn, model = pretrain(train_x, train_y, hp)

    # advance the training loop with model hyper-parameters
        @printf("  ****** Advancing model %d iterations\n", iters)
        hp.epochs = iters  # set iters to train before grad testing
        stats = setup_stats(hp, dotest)
        training_time = training_loop!(hp, train, mb, nnw, bn, stats, model)
        output_stats(train, nnw, hp, bn, training_time, stats, model)


    # set new hyper-parameters for gradient checking
    if hp.do_batch_norm
        minihp = hp  # leave hyper-parameters as is
    else
        minihp = set_hp(hp)
        setup_functions!(model, minihp, train.out_k)  
        @bp
    end

    return minihp, nnw, train, bn, model 
end


# method to use full model with real training data
function check_grads(hp, train_x, train_y, iters; samplepct=.005, quiet=false, tweak=1e-6, post_iters=0,
    biasidx = [], thetaidx = [])

    println("  ****** Testing gradient calculation on input training dataset")
    println("  ****** Setting up model and data")

    @bp
    hp, nnw, train, bn, model = prep_check(hp, train_x, train_y, iters; samplepct=samplepct, quiet=quiet, tweak=tweak)

    run_check_grads(hp, nnw, train, model; bn=bn, samplepct=samplepct, tweak=tweak, quiet=quiet, iters=post_iters,
                    biasgradidx = biasidx, thetagradidx = thetaidx)
end


# method to use mini-model with same number/type of hidden layers as designed model
function prep_check(hp; in_k=3, out_k=3, m=5, hid=5, iters=30, samplepct=1.0, tweak=1e-4, quiet=false)

    # set new hyper-parameters for gradient checking
    minihp = set_hp(hp)

    # create minidata and mini-model
    # @bp
    minidat, minimb, miniwgts, minibn, ministats, minimodel = use_minidata(minihp;
        in_k=in_k, out_k=out_k, m=m, hid=hid)   

    # advance some iterations using the analytic model
    @printf("\n  ****** Advancing model %d iterations\n", iters)
    minihp.epochs = iters

    training_time = training_loop!(minihp, minidat, minimb, miniwgts, minibn, ministats, minimodel)

    # TODO do we want this?  minihp and setup_stats currently setup so that no training stats get collected
    quiet && output_stats(minidat, miniwgts, minihp, training_time, ministats, minimodel)

    return minihp, miniwgts, minidat, minimodel
end


# method to use mini-model with same number/type of hidden layers as designed model
function check_grads(hp; in_k=3, out_k=3, m=5, hid=5, iters=30, samplepct=1.0, tweak=1e-4, quiet=false,
    post_iters=0, biasidx = [], thetaidx = [])
    println("  ****** Testing gradient calculation on minitiature dataset")

    println("  ****** Setting up model and test data")

    # @bp
    minihp, miniwgts, minidat, minimodel = prep_check(hp; in_k=in_k, out_k=out_k, m=m, hid=hid, iters=iters,
        samplepct=samplepct, tweak=tweak, quiet=false)

    # @bp
    run_check_grads(minihp, miniwgts, minidat, minimodel; samplepct=samplepct, tweak=tweak, quiet=quiet)
end


# eps_wgt = (lr, idx, typ) where lr is int of layer (either hidden or output) 
#                                idx is the linear index of theta or bias for lr
#                                typ must be "theta" or "bias"
function check_one(hp, inwgts, indat, model; tweak=1e-6, eps_wgt=(2,3,"bias"), example=10)  
    wgts = deepcopy(inwgts)
    dat = deepcopy(indat)
    selr = eps_wgt[1]
    idx = eps_wgt[2]
    typ = eps_wgt[3]
    n_layers = wgts.output_layer
    deltype = if typ == "theta" 
                    "delta_th" 
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
    feedfwd!(onedat, wgts, hp, bn, model.ff_execstack, dotrain=false) 
    cost = model.cost_function(onedat.targets[:,1], onedat.a[wgts.output_layer][:,1], 1, wgts.theta, hp.lambda, hp.reg,
                          wgts.output_layer)
    
    # 2. compute the gradient of the weight we will perturb by tweak (not yet...)
    backprop!(wgts, dat, hp)  # for all layers   
    gradone = getproperty(wgts, Symbol(deltype))[selr][idx]  # delta_th or delta_b

    # 3. calculate the loss for the perturbed weight
    w = getproperty(wgts, Symbol(typ))[selr][idx]  # bias or theta
    w_plus = w + tweak
    getproperty(wgts, Symbol(typ))[selr][idx] = w_plus  # set the chosen weight to tweaked value
    feedfwd!(onedat, wgts, hp, bn, model.ff_execstack, dotrain=false) 
    # cost_plus = model.cost_function(onedat.targets[:,1], onedat.a[wgts.output_layer][:,1], 1, wgts.theta, hp.lambda, hp.reg,
                          # wgts.output_layer)
    costnew = model.cost_function(onedat.targets[:,1], onedat.a[wgts.output_layer][:,1], 1, wgts.theta, hp.lambda, hp.reg,
                          wgts.output_layer)

    # w_minus = w - tweak
    # getproperty(wgts, Symbol(typ))[selr][idx] = w_minus  # set the chosen weight to tweaked value
    # feedfwd_predict!(onedat, wgts, hp) 
    # cost_minus = model.cost_function(onedat.targets[:,1], onedat.a[wgts.output_layer][:,1], 1, wgts.theta, hp.lambda, hp.reg,
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


function run_check_grads(hp, wgts, dat, model; bn=Batch_norm_params(), samplepct=.015, tweak=1e-6, quiet=false, iters=0,
    biasgradidx = [], thetagradidx = [])  # last 2 args are the indices of the bias and theta we sample

    println("\n\n ******   Starting run_check_grads")
    printstruct(hp)
    println()

    if !(samplepct > 0.0 && samplepct <= 1.0)
        error("input samplepct must be greater than zero and less than or equal to 1.0")
    end

    if length(biasgradidx) == 0  # no input of pre-selected indices
        if samplepct == 1.0
            biasgradidx = [eachindex(x) for x in wgts.bias[2:wgts.output_layer]]
        else  # must be in (0.0,1.0)
            biasgradidx = [sample_idx(samplepct,x) for x in wgts.bias[2:wgts.output_layer]]
        end
        insert!(biasgradidx, 1, eachindex([]))  # we need this placeholder for layer 1
    end

    if length(thetagradidx) == 0  # no input of pre-selected indices
        if samplepct == 1.0
            thetagradidx = [eachindex(x) for x in wgts.theta[2:wgts.output_layer]]
        else  # must be in (0.0,1.0)
            thetagradidx = [sample_idx(samplepct,x) for x in wgts.theta[2:wgts.output_layer]]
        end
        insert!(thetagradidx, 1, eachindex([]))  # we need this placeholder for layer 1
    end
    
    # run model a few more iterations:  especially valuable if full model uses minibatches
    # TODO  does this work or mess things up?
    for i = 1:iters
        feedfwd!(dat, wgts, hp, bn, model.ff_execstack, dotrain=false) # no batches, no batchnorm, no dropout
        backprop!(wgts, dat, hp)
        update_parameters!(wgts, hp)
    end

    # initialize numeric gradients results holder
    numgradtheta = [zeros(length(x)) for x in thetagradidx]  
    numgradbias  = [zeros(length(x)) for x in biasgradidx]  
    println("numgradbias holder ", size(numgradbias), " ", size(numgradbias[1]))

    # compute numgrad
    kinkrejecttheta, kinkrejectbias, numcost,  numpreds, bnnumgrads = compute_numgrad!(numgradtheta, numgradbias, 
                        thetagradidx, biasgradidx, wgts, dat, hp, model; bn=bn, tweak=tweak)
    numgrad = (numgradtheta,numgradbias)

    # compute model grad
    println("  ****** Calculating feedfwd/backprop gradients")
    # @bp
    modcost, modpreds, bnmodgrads = compute_modelgrad!(dat, wgts, hp, bn, model)  # changes dat, wgts and hp
    modgrad = (deepcopy(wgts.delta_th), deepcopy(wgts.delta_b))  # do I need to copy since compute_numgrad! does no backprop?

    
    # compare results from modgrads and numgrads
     println("  ****** Are costs the same? (model, num cost) ")
     println(modcost, " ", numcost)
     println("  ****** How close are the predictions?")
     println(count(modpreds .== numpreds), " of ", length(modpreds))
     println("      mean abs difference ",sum(abs.(modpreds .- numpreds)) / length(modpreds))

    # filter modgrads to match sample of numgrads
    modgrad = [   [modgrad[1][i][thetagradidx[i]] for i in 2:wgts.output_layer],     
                  [modgrad[2][i][biasgradidx[i]] for i in 2:wgts.output_layer]   ]

    println("modgrad output ", size(modgrad), " ", size(modgrad[2]))
    println("numgrad output ", length(numgrad), " ", size(numgrad[2]))
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

        # print batchnorm gradients

        if hp.do_batch_norm
            bnmodgrads = [   [bnmodgrads[1][i][biasgradidx[i]] for i in 2:wgts.output_layer-1],     
                          [bnmodgrads[2][i][biasgradidx[i]] for i in 2:wgts.output_layer-1]   ]

            # println("bnmodgrads ", size(bnmodgrads), " ", size(bnmodgrads[2]), " ", size(bnmodgrads[2][2]))
            # println("bnnumgrads ", size(bnnumgrads), " ", size(bnnumgrads[2]), " ", size(bnnumgrads[2][2]))
            bncols = hcat(flat(bnmodgrads), flat(bnnumgrads))
            println("batchnorm parameter gradients\n")
            printby2(bncols)            
        else 
            # do nothing
        end
    end  # begin block

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


"""
two methods for dealing with batch normalization
1.  do a final feedfwd_predict using the batch norm terms. Then feedfwd and backprop to generate theta
    and bias weights without batchnorm terms: mu, stddev, gamma and beta. Possibly more than one 
    "settle down" iteration will be needed. Then:  
    For analytic, backprop to calculate the deltas for the weights, leaving out the batchnorm terms. 
    For numeric, calculate numeric approximation for weights, leaving out the batchnorm terms.
    This can work, but the high learning rate possible with batchnorm won't work in many cases. 
    The bias weights will be off.

2.  Include the gamma and beta parameters and compare numeric approximation and backprop deltas for theta, gamma
    and delta.  Use running average of mu and stddev for feedforward, as one would for a test or validation
    data set.  
    For analytic, backprop to calculate the delta updates for gamma and delta.
    For numeric, perturb (separately) gamma and beta and use cost predictions to calculate the approximations to 
    the deltas.
    This method is better because it checks are code to see if we calculate the deltas for all trained parameters.

"""


function set_hp(hp)
    # set appropriate bogus hyper-parameters for mini-model AND FOR FINAL GRAD CALCULATION RUN???
    minihp = Hyper_parameters()

        # change parameters to special values for grad checking
        minihp.reg = ""                # no reg
        # minihp.stats = []              # no stats    STATS ARE OK. DON'T HURT ANYTHING OR COST THAT MUCH IN PERF FOR THIS
        minihp.plot_now = false        # no plots
        minihp.opt = "none"            # no optimization
        minihp.learn_decay = [1.0, 1.0]  # no optimization
        minihp.do_batch_norm = false   # no batches
        minihp.dobatch = false         # no batches
        minihp.alphamod = 0.02             # low learning rate
        minihp.dropout = false
        minihp.norm_mode = "minmax"  # only used with minimodel
        minihp.bias_initializer = hp.bias_initializer  # only used with minimodel
        minihp.lambda = 0.0          # shouldn't matter as reg is ""
        minihp.sparse = false
        minihp.quiet = true

        # use the real models params here
        minihp.hidden = hp.hidden
        minihp.classify = hp.classify

        # printstruct(minihp)

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
    minidat, minimb, miniwgts, minibn, minimodel = pretrain(mini_x, mini_y, minihp)

    # repeatable initization of theta and bias
    for l = 2:miniwgts.output_layer
        # debug_initialize_weights!(miniwgts.theta[l])
        xavier_initialize_weights!(miniwgts.theta[l])
        bias_val!(miniwgts.bias[l], minihp.bias_initializer)
    end

    dotest = false
    ministats = setup_stats(minihp, dotest)  

    return minidat, minimb, miniwgts, minibn, ministats, minimodel
end

"""
caution 
Kinks in the objective. One source of inaccuracy to be aware of during gradient checking is the problem of kinks. Kinks refer to non-differentiable parts of an objective function, introduced by functions such as ReLU (max(0,x)), or the SVM loss, Maxout neurons, etc. Consider gradient checking the ReLU function at x=−1e6. Since x<0, the analytic gradient at this point is exactly zero. However, the numerical gradient would suddenly compute a non-zero gradient because f(x+h) might cross over the kink (e.g. if h>1e−6) and introduce a non-zero contribution. You might think that this is a pathological case, but in fact this case can be very common. For example, an SVM for CIFAR-10 contains up to 450,000 max(0,x) terms because there are 50,000 examples and each example yields 9 terms to the objective. Moreover, a Neural Network with an SVM classifier will contain many more kinks due to ReLUs.

Note that it is possible to know if a kink was crossed in the evaluation of the loss. This can be done by keeping track of the identities of all “winners” in a function of form max(x,y); That is, was x or y higher during the forward pass. If the identity of at least one winner changes when evaluating f(x+h) and then f(x−h), then a kink was crossed and the numerical gradient will not be exact.
"""


function compute_numgrad!(numgradtheta, numgradbias, thetagradidx, biasgradidx, wgts, dat, hp, model; 
    bn=Batch_norm_params(), tweak=1e-7, quiet=true)

#   TODO how to handle conditional passing of bn struct
#   TODO factor this to handle 1 pass of numeric grads.  Caller builds the complete set.
#   TODO be sure we use right hp values and setup functions to match

    # @bp


    kinkcnt = 0
    kinkrejectbias = Tuple{Int64, Int64}[]
    kinkrejecttheta = Tuple{Int64, Int64}[]
    comploss = 0.0
    numpreds = deepcopy(dat.a[wgts.output_layer])

    olddat = deepcopy(dat)
    output_layer = wgts.output_layer


    # test first pass of cost

        # feedfwd_predict!(dat, wgts, hp)   
        # loss1 = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n)  
        # println("1st pass at cost ", loss1)   

    for lr in 2:wgts.output_layer  # loop by layer
        @printf("  ****** Calculating numeric bias gradients layer: %d\n", lr)

        for (idx,bi) in enumerate(biasgradidx[lr])   # eachindex(wgts.bias[lr])  # tweak each bias, compute partial diff for each bias

            wgts.bias[lr][bi] += tweak

            feedfwd!(dat, wgts, hp, bn, model.ff_execstack, dotrain=false)       #feedfwd_predict!(dat, wgts, hp)
            kinktst1 = findkink(olddat, dat, output_layer)
            loss1 = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) 

            wgts.bias[lr][bi] -= 2.0 * tweak

            feedfwd!(dat, wgts, hp, bn, model.ff_execstack, dotrain=false)
            kinktst2 = findkink(olddat, dat, output_layer)
            loss2 = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n)   

            wgts.bias[lr][bi] += tweak   

            if kinktst1  || kinktst2 
                kinkcnt += 1
                push!(kinkrejectbias, (lr, bi))
            end

            numgradbias[lr][idx] = (loss1 - loss2) / (2.0 * tweak)  

            !quiet &&  @printf("loss1: %f loss2: %f grad: %f kink: %d\n", loss1, loss2, numgradbias[lr][idx], kinkcnt)

        end

        println("kink errors for bias = ", kinkcnt)

        @printf("  ****** Calculating numeric theta gradients layer: %d\n", lr)

        for (idx,thi) in enumerate(thetagradidx[lr])  # eachindex(wgts.theta[lr])  # tweak each theta, compute partial diff for each theta

            wgts.theta[lr][thi] += tweak

            feedfwd!(dat, wgts, hp, bn, model.ff_execstack, dotrain=false)
            kinktst1 = findkink(olddat, dat, output_layer)
            loss1 = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) 

            wgts.theta[lr][thi] -= 2.0 * tweak

            feedfwd!(dat, wgts, hp, bn, model.ff_execstack, dotrain=false)
            kinktst2 = findkink(olddat, dat, output_layer)
            loss2 = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n)

            wgts.theta[lr][thi] += tweak   # set it back to starting value
            comploss = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) # diagnostic

            if kinktst1 || kinktst2 
                kinkcnt += 1
                push!(kinkrejecttheta, (lr, thi))
            end

            numgradtheta[lr][idx] = (loss1 - loss2) / (2.0 * tweak)

            !quiet &&  @printf("loss1: %f loss2: %f grad: %f kink: %d\n", loss1, loss2, numgradtheta[lr][idx], kinkcnt)

            numpreds = dat.a[wgts.output_layer]
        end
    end
    println("count of gradients rejected: ", kinkcnt)


    if hp.do_batch_norm
        gamgrad  = [zeros(length(x)) for x in biasgradidx]
        betgrad  = [zeros(length(x)) for x in biasgradidx]

        for lr = 2:wgts.output_layer-1

            @printf("  ****** Calculating numeric batchnorm gamma gradients layer: %d\n", lr)
            for (idx,bi) in enumerate(biasgradidx[lr]) 
                println(idx)
                bn.gam[lr][bi] += tweak
                feedfwd!(dat, wgts, hp, bn, model.ff_execstack, dotrain=false)  # bn is closure in function placeholder
                loss1 = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) 
                bn.gam[lr][bi] -= 2.0 * tweak
                feedfwd!(dat, wgts, hp, bn, model.ff_execstack, dotrain=false)  # bn is closure in function placeholder
                loss2 = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) 
                bn.gam[lr][bi] += tweak
                gamgrad[lr][idx] = (loss1 - loss2) / (2.0 * tweak)
            end

            @printf("  ****** Calculating numeric batchnorm beta gradients layer: %d\n", lr)
            for (idx,bi) in enumerate(biasgradidx[lr])
                bn.bet[lr][bi] += tweak
                feedfwd!(dat, wgts, hp, bn, model.ff_execstack, dotrain=false)  # bn is closure in function placeholder
                loss1 = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) 
                bn.bet[lr][bi] -= 2.0 * tweak
                feedfwd!(dat, wgts, hp, bn, model.ff_execstack, dotrain=false)  # bn is closure in function placeholder
                loss2 = model.cost_function(dat.targets, dat.a[wgts.output_layer], dat.n) 
                bn.bet[lr][bi] += tweak
                betgrad[lr][idx] = (loss1 - loss2) / (2.0 * tweak)
            end
        end
        bngrads = [gamgrad[2:wgts.output_layer-1], betgrad[2:wgts.output_layer-1]]
        println("bngrads ", size(bngrads), " ", size(bngrads[1]), " ", size(bngrads[2]))

    else
        bngrads = [[], []]  # signifies empty
    end

    return kinkrejecttheta, kinkrejectbias, comploss, numpreds, bngrads
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


function compute_modelgrad!(dat, nnw, hp, bn, model)

    # @bp

    cost, accuracy = quick_stats(dat, nnw, hp, model)
    println("accuracy: ", accuracy)
    println("cost: ", cost)

    # @bp
    hp.mb_size = float(dat.n)
    feedfwd!(dat, nnw, hp, bn, model.ff_execstack, dotrain=false)  
    backprop!(nnw, dat, hp)  # for all layers   

    if hp.do_batch_norm
        for lr in 2:nnw.output_layer-1  # defaults never touched; set to zero for comparison to numeric grads
            nnw.delta_b[lr] .= 0.0
        end
        bngrads = [bn.delta_gam, bn.delta_bet]
        println("model bn grads ", size(bngrads), " ", size(bngrads[2]), " ", size(bngrads[2][2]))
    else
        bngrads = [[] for i in 2:nnw.output_layer-1]
    end

    return cost, dat.a[nnw.output_layer], bngrads
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


# method works on the number of elements you want to sample from
function sample_idx(xpct, lx::Int)
    xcnt = ceil(Int, xpct * lx) + 2
    sort!(randperm(lx)[1:xcnt])
end


# method works on the object that you want indices for
function sample_idx(xpct, x::Array{T}) where {T}
    lx = length(x)
    sample_idx(xpct, lx)
end


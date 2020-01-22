# maybe do the funcs and args for a layer at the same time--same tests drive each
# how much to do at once:  a single function or a layer of n functions?

using GeneralNN


function test_run(tomlfn, datafn, do_mb = false)
    hp = GeneralNN.setup_params(tomlfn)
    trainx, trainy = GeneralNN.extract_data(datafn)
    GeneralNN.shuffle_data!(trainx, trainy)
    train, mb, nnw, bn = GeneralNN.pretrain(trainx, trainy, hp)


    func_dict = create_funcs(train, nnw, bn, hp)
    n_layers = length(hp.hidden) + 2
    strstack = build_string_stack(hp)
    execstack = build_exec_stack(strstack, func_dict)

    if do_mb 
        GeneralNN.update_batch_views!(mb, train, nnw, hp, 1:50) 
        dat = mb
    else
        dat = train
    end
    model_runner!(dat, nnw, bn, execstack, n_layers)

    ret = Dict(
            "train_inputs" => trainx, 
            "train_targets"=> trainy, 
            "train_preds" => dat.a[nnw.output_layer], 
            "Wgts" => nnw, 
            "batchnorm_params" => bn, 
            "hyper_params" => hp,
            "strstack" => strstack,
            "execstack"  => execstack
            )
end


#   builds the strings for functions and arguments
function build_string_stack(hp)
    strstack = []  # will be array of string arrays, 1 per layer group

    n_hid = length(hp.hidden)
    # input layer
    i = 1
        push!(strstack, String[])
        if hp.dropout && (hp.droplim[1] < 1.0)
            push!(strstack[i], "dropout")
        end

    # hidden layers        
    for i = 2:n_hid+1 
        push!(strstack, String[])

        # affine either or...
        hp.do_batch_norm && push!(strstack[i], "affine_nobias") # true
        hp.do_batch_norm || push!(strstack[i], "affine") # false
        # batch_norm_fwd 
        hp.do_batch_norm && push!(strstack[i], "batch_norm_fwd")
        # unit_function --> updates in place
        unit_function = 
            if hp.hidden[i-1][1] == "sigmoid"
                "sigmoid"
            elseif hp.hidden[i-1][1] == "l_relu"
                "l_relu"
            elseif hp.hidden[i-1][1] == "relu"
                "relu"
            elseif hp.hidden[i-1][1] == "tanh"
                "tanh_act"
            end
        push!(strstack[i], unit_function)
        # dropout --> updates in place
        if hp.dropout && (hp.droplim[hl] < 1.0)
            push!(strstack[i], "dropout")
        end        
    end

    #   output layer
    i = n_hid + 2
        push!(strstack, String[])

        push!(strstack[i], "affine")
        # classify_function --> updates in place
        classify_function = 
            # if train.out_k > 1  # more than one output (unit)
                if hp.classify == "sigmoid"
                    "sigmoid"
                elseif hp.classify == "softmax"
                    "softmax"
                else
                    error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
                end
            # else
            #     if hp.classify == "sigmoid" || hp.classify == "logistic"
            #         "logistic!"  # for one output label
            #     elseif hp.classify == "regression"
            #         "regression!"
            #     else
            #         error("Function to classify output must be \"sigmoid\", \"logistic\" or \"regression\".")
            #     end
            # end
        push!(strstack[i], classify_function)

    return strstack
end


function create_funcs(dat, nnw, bn, hp)
    # curried function definitions
    affine!(hl) = GeneralNN.affine!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
    affine_nobias!(hl) = GeneralNN.affine_nobias!(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])
    relu!(hl) = GeneralNN.relu!(dat.a[hl], dat.z[hl])
    softmax!(hl) = GeneralNN.softmax!(dat.a[hl], dat.z[hl])
    batch_norm_fwd!(hl) = GeneralNN.batch_norm_fwd!(dat, bn, hp, hl)

    # indexable function container
    func_dict = Dict(
                        "affine" => affine!,
                        "affine_nobias" => affine_nobias!,
                        "relu" => relu!,
                        "softmax" => softmax!,
                        "batch_norm_fwd" => batch_norm_fwd!
                    )
end


function build_exec_stack(strstack, func_dict)  
    execstack = []
    for i in 1:size(strstack,1)
        push!(execstack,[])
        for r in strstack[i]
            push!(execstack[i], func_dict[r])
        end
    end
    return execstack
end


function model_runner!(dat, nnw, bn, execstack, n_layers)
    for hl in 1:n_layers
        layer_grp = execstack[hl]
        for f in layer_grp
            f(hl)
        end
    end
end


function build_ff_layer(ff_funcs, ff_args)

    # function selection by:
        # dispatch on argument list or
        # use explicit function names

    # loop setup
        # set_minibatch_size --> returns values
        # update_minibatch_views --> updates in place
            # update_minibatch_views for dropout

    # feed forward
        # input_layer
            # dropout
        # hidden layers: HOW TO PUT RIGHT HIDDEN LAYER IN ARGUMENTS
            # dropout --> updates in place
            # affine_w_bias --> updates in place
            # affine_no_bias --> updates in place
            # batch_norm_fwd --> updates in place
            # unit_function --> updates in place

        # output layer
            # affine  --> updates in place
            # classify_function --> updates in place

    # backprop
        # output layer returns epsilon, delta_th, delta_bias (sometimes)
        # hidden layers
            # gradient via gradient_function!  (updates in place)
            # epsilon  (returns array)
            # dropout updates layers epsilon with filter
            # batch_norm_back updates delta_th, delta for batch_norm parameters
            # return delta_th or
            # return both delta_th, delta_b

    # optimization

    # update parameters

    # returns a list or 
    # returns a dict of functions--this could be pre-built and...
        # if returning a dict, returns list of dict keys for functions to actually run
end


function build_a_layer_args()
# returns a list or dict of the args for a layer's functions
    # the args for a given function are a tuple

end


function build_ff_predict_layers()
end




function build_ff_train_args(hp,nnw)
end


function run_loop()
    # put function and arguments together
    # can do this if all of the variables exists with args as tuples
    #     or can do with strings by wrapping the string as    eval(Meta.parse("string"))
end


function build_all_funcs() # later replace with 1 layer at a time
    all_funcs = Dict{String,Function}()
    # layer functions
    all_funcs["affine"] = GeneralNN.affine! 
    all_funcs["affine_nobias"] = GeneralNN.affine_nobias!
    all_funcs["sigmoid"] = GeneralNN.sigmoid! 
    all_funcs["tanh_act"] = GeneralNN.tanh_act! 
    all_funcs["l_relu"] = GeneralNN.l_relu!
    all_funcs["relu"] = GeneralNN.relu!
    # classifiers 
    all_funcs["softmax"] = GeneralNN.softmax! 
    all_funcs["logistic"] = GeneralNN.logistic!
    all_funcs["regression"] = GeneralNN.regression!
    # gradients
    all_funcs["sigmoid_gradient"] = GeneralNN.sigmoid_gradient!
    all_funcs["tanh_act_gradient"] = GeneralNN.tanh_act_gradient!
    all_funcs["l_relu_gradient"] = GeneralNN.l_relu_gradient!
    all_funcs["relu_gradient"] = GeneralNN.relu_gradient!
    # others
    all_funcs["dropout"] = GeneralNN.dropout_fwd!
    all_funcs["batch_norm_fwd"] = GeneralNN.batch_norm_fwd!

    return all_funcs
end


# function build_all_args()
# # these are the actual argument values used to call the functions,
# #    NOT the parameters/arguments used in the function signatures
#     all_args = Dict{String,String}()
#     # args for layer functions
#     all_args["affine"] = "(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])"
#     all_args["affine_nobias"] = "(dat.z[hl], dat.a[hl-1], nnw.theta[hl], nnw.bias[hl])"  # could use specific list ultimately
#     all_args["batch_norm_fwd"] = "(dat, bn, hp, hl)"
#     all_args["sigmoid"] = "(dat.a[hl], dat.z[hl])"
#     all_args["tanh_act"] = "(dat.a[hl], dat.z[hl])"
#     all_args["tanh_act"] = "(dat.a[hl], dat.z[hl])"
#     all_args["l_relu"] = "(dat.a[hl], dat.z[hl])"
#     all_args["relu"] = "(dat.a[hl], dat.z[hl])"
#     # args for classifiers
#     all_args["softmax"] = "(dat.a[nnw.output_layer], dat.z[nnw.output_layer])"
#     all_args["logistic"] = "(dat.a[nnw.output_layer], dat.z[nnw.output_layer])"
#     all_args["regression"] = "(dat.a[nnw.output_layer], dat.z[nnw.output_layer])"
#     # args for gradients
#     all_args["sigmoid_gradient"] = "(dat.grad[hl], dat.z[hl])"
#     all_args["tanh_act_gradient"] = "(dat.grad[hl], dat.z[hl])"
#     all_args["l_relu_gradient"] = "(dat.grad[hl], dat.z[hl])"
#     all_args["relu_gradient"] = "(dat.grad[hl], dat.z[hl])"
#     # args for others
#     all_args["dropout"] = ""

#     return all_args
# end


function build_all_args(dat, nnw, hp, bn)
# these are the actual argument values used to call the functions,
#    NOT the parameters/arguments used in the function signatures
    all_args = Dict{String,String}()
    # args for layer functions
    all_args["affine"] = (dat.z[:hl], dat.a[:hl-1], nnw.theta[:hl], nnw.bias[:hl])
    all_args["affine_nobias"] = (dat.z[:hl], dat.a[:hl-1], nnw.theta[:hl], nnw.bias[:hl])  # could use specific list ultimately
    all_args["batch_norm_fwd"] = (dat, bn, hp, :hl)
    all_args["sigmoid"] = (dat.a[:hl], dat.z[:hl])
    all_args["tanh_act"] = (dat.a[:hl], dat.z[:hl])
    all_args["tanh_act"] = (dat.a[:hl], dat.z[:hl])
    all_args["l_relu"] = (dat.a[:hl], dat.z[:hl])
    all_args["relu"] = (dat.a[:hl], dat.z[:hl])
    # args for classifiers
    all_args["softmax"] = (dat.a[nnw.output_layer], dat.z[nnw.output_layer])
    all_args["logistic"] = (dat.a[nnw.output_layer], dat.z[nnw.output_layer])
    all_args["regression"] = (dat.a[nnw.output_layer], dat.z[nnw.output_layer])
    # args for gradients
    all_args["sigmoid_gradient"] = (dat.grad[:hl], dat.z[:hl])
    all_args["tanh_act_gradient"] = (dat.grad[:hl], dat.z[:hl])
    all_args["l_relu_gradient"] = (dat.grad[:hl], dat.z[:hl])
    all_args["relu_gradient"] = (dat.grad[:hl], dat.z[:hl])
    # args for others
    all_args["dropout"] = ""

    return all_args
end

####################################################################
# simple test case for this approach
####################################################################

# function affine!(z,a,theta)
#     z[:] = theta * a
# end

# function affine!(z,a,theta,bias)
#     z[:] = theta *  a .+ bias
# end

# function sigmoid!(a,z)
#     a[:] = 1.0 ./ (1.0 .+ exp.(.-z))  
# end

# function relu!(a,z)
#     a[:] = max.(z, 0.0)
# end

function build_test_data()

    # linear
    z = []
    push!(z, zeros(5,8))
    push!(z,zeros(5,8))
    push!(z,zeros(5,8))

    # activations
    a = []
    push!(a,rand(5,8))
    push!(a,rand(5,8))
    push!(a,rand(5,8))

    theta = []
    push!(theta,rand(5,5))
    push!(theta,rand(5,5))
    push!(theta,rand(5,5))

    bias = []
    push!(bias, fill(0.2,(5)))
    push!(bias, fill(0.2,(5)))
    push!(bias, fill(0.2,(5)))

    return z, a, theta, bias
end


# first test:  hard code the functions and function list--e.g., no building

# all_funcs = Dict("affine!" => affine!, "sigmoid!" => sigmoid!, "relu!" => relu!)
# all_args = Dict("affine w bias" => "(z[#i],a[#i-1],theta[#i],bias[#i])", "affine no bias" => "(z[#i], a[#i - 1], theta[#i])", 
#             "sigmoid!" => "(a[#i],z[#i])", "relu!" => "(a[#i],z[#i])")




function build_layer(func, arg)
    return (func, arg)  # Meta.parse(arg)
end


function runtst(tomlfn, datafn)
    hp = GeneralNN.setup_params(tomlfn)
    trainx, trainy = GeneralNN.extract_data(datafn)
    train, mb, nnw, bn = GeneralNN.pretrain(trainx, trainy, hp)

    for i = 1:4
        println(tst(dat, i))
    end
end

function tst(dat, hl)
    dat.z[hl][1]
end

j = rand(5)
k = zeros(5)
macro runit(op, ins)
    return quote
                local args = Meta.parse.($ins)
                $op(eval(args.args[1]), eval(args.args[2]))
           end
end




# this will be batch_norm faster than an explicit eval()
macro get_arg(argument)
    quote
        eval($(esc(argument)))
    end
end

# this puts the right value in for the layer number
#    requires parsing and eval when running the queue
function builder_old(func1, arg1, func2, arg2)
    dofuncs = []
    doargs = []
    for i in 2:3
        push!(dofuncs, all_funcs[func1])
        push!(doargs, replace(all_args[arg1], "#i" => i))
        push!(dofuncs,all_funcs[func2])
        push!(doargs, replace(all_args[arg2], "#i" => i))
    end
    return dofuncs, doargs
end

# this actually works: parses and evals every time the queue is run
function runner_old(funclist, arglist)
    for (i,j) in zip(funclist, arglist)
        i(eval(Meta.parse(j))...)
        # println("Did $i($j...)")
    end
end


# not using...
function test_new_builder(hp, func_dict)
    # dat, mb, nnw, bn = pretrain(trainx, trainy, hp)

    # build model stack
    strstack = build_string_stack(hp)
    execstack = build_exec_stack(strstack, func_dict)

end



###################################################
# type experiments
###################################################

"""
julia> supertype(Tvz)
DenseArray{SubArray{Float64,2,SparseMatrixCSC{Float64,Int64},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},false},1}
Tvz
Array{SubArray{Float64,2,SparseMatrixCSC{Float64,Int64},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},false},1}



julia> supertype(Tva)
DenseArray{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
Tva
Array{SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64}},true},1}
"""
### Training Loop

#### Feed Forward

##### Feed forward execstack loop

**How arguments are provided to each function called in the loop**

The execstack loop calls each layer function used during the feed forward process of training. These functions come from the model.ff_execstack, which is an array which contains arrays of functions, each of which we will call a layer_group.  Each layer_group is an array of functions: simply the functions to be called in each layer of the model.  For this example we will use relu!. 

The target layer function, for example relu!, is set to the alias f because the loop always uses f as the name of functions to be called. So, the value of f will be set to the function relu! in one iteration of the loop for a specific hidden layer. Here is how the arguments to the target function are set in the loop that runs all of the functions.

Because each target function to be called may require different inputs, we set the inputs as the output of the helper function argset().  Argset has a method for each of the various functions that may be called during feed forward. These are setup in advance so that the appropriate argset method returns a tuple of the inputs to a specific target function. Here is how it works for the example target function relu!().

The caller of, feedfwd! is called with these inputs: `(dat, nnw, hp, bn, model.ff_execstack)`.

The feedfwd! function signature is: `(dat::Union{Batch_view,Model_data}, nnw, hp, bn, ff_execstack)` so these variable names are available within the body of feedfwd!.

f, or relu!, is called like this: `f(argset(dat, nnw, hp, bn, lr, f)...)`and this also shows how argset is called. Note that lr is the loop counter that corresponds to each layer group and f is just the alias to the target function,

The argset function signature is the same for every method and defines the inputs that are used to create the potentially different inputs that are passed on to each of the target functions, as needed for each target function.  The function signature is: `(dat::Union{Model_data, Batch_view}, nnw::Wgts, hp::Hyper_parameters, bn::Batch_norm_params, hl::Int, fn::typeof(relu!))`. Note that the type of the last input parameter is unique to each target function, which enables method dispatch to select and run the right method specific to each target function. So, dat is bound to dat, nnw to nnw, hp to hp, bn to bn, lr to hl, and f to fn.

The output of this argset method is: `dat.a[hl], dat.z[hl]`, which are the inputs with which f, namely relu!, is called.

Finally, the input signature to the target function, relu!, is: `(a, z)`. So, dat.a[hl] -- member a layer hl of the object dat -- is bound to a, and dat.z[hl] is bound to z.
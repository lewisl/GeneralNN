## Redesign Notes

### Basics
Factor the overall approach into:

- defining the model
- describing and accessing the input data
- training: specifying training parameters: params
- training: prepping and running the process: train(data, model, params)

### layer API

#### per layer description
	unit_type or activation_function
	activation_gradient_function
	no_of_units
	fully_connected
	if not fully_connected, inputs
	optimization
	regularization (as the cost partial derivative is applied to the layer)
	do_affine_bool (if not treating affine as a separate layer)
    use_bias_bool

        Look at the sizing calculations to see what inputs need to be specified. Must specify number of hidden units. Should not need to specify anything about input layer size as an input training array must be provided. For output, decide if one-hot encoding must be done before starting the framework.

#### specific to input layer
	no_of_samples = this can be determined from input training data
	no_of_units = no_of_features = this can be determined from input training data
	normalization_bool -> part of rundefs
	normalization_method or normalization_function -> part of rundefs
	randomize_sample_order or shuffle -> part of rundefs

#### specific to output layer
	classification_function (normally takes the place of an activation function, which could be done in previous layer)
	or call it classifier
	special handling for one-v-all or one-v-?
	accuracy_function (for accuracy metric)
	cost_function
	cost_gradient_function -> determined from cost_function
	regularization (as the penalty is applied to the cost function) -> part of rundefs

### Structural changes
    layer descriptors by layer
    some hyper parameters by layer->make a list
    dataflow connections: 
        default is implicit for all fully connected layers
        provide a way to make it explicit
    distinguish using built-in functions vs user-provided functions
    each spec describes all layers vs. describe by layer for all specs
        former is more compact but needs more elaborate error checking and "building"
        latter is more verbose but less error prone--verboseness reduced with "ditto"

### Example 1

```json
{
    "layerdefs": {
        "layer2": {
            "n_hid": 80,
            "units": "relu"
        },
        "layer1": {
            "input": 9999999999999999
        },
        "layer3": {
            "classify": "softmax"
        }
    },
    "rundefs": {
        "epochs":  24,
        "alpha":   0.74,
        "reg":  "Maxnorm",
        "maxnorm_lim": [4.0],
        "lambda":  0.000191,
        "learn_decay": [0.5,4.0],
        "mb_size_in":   50, 
        "norm_mode":   "none",
        "dobatch": true,
        "do_batch_norm":  true,
        "opt":   "adam",
        "opt_params": [0.9, 0.999],
        "dropout": false,
        "droplim": [1.0,0.8,1.0],
        "plots": ["Train", "Learning", "Test"],
        "plot_now": true,
        "quiet": true,
        "initializer": "xavier"
    }
}
```

### Example 2

```json
{
    "layerdefs": [
        {   "layer": 2,
            "n_hid": 80,
            "units": "relu"
        },
        {   "layer": 1,
            "input": "??????????"
        },
        {   "layer": 3, 
            "output": "??????????",
            "classify": "softmax"
        }
    ],
    "rundefs": {
        "epochs":  24,
        "alpha":   0.74,
        "reg":  "Maxnorm",
        "maxnorm_lim": [4.0],
        "lambda":  0.000191,
        "learn_decay": [0.5,4.0],
        "mb_size_in":   50, 
        "norm_mode":   "none",
        "dobatch": true,
        "do_batch_norm":  true,
        "opt":   "adam",
        "opt_params": [0.9, 0.999],
        "dropout": false,
        "droplim": [1.0,0.8,1.0],
        "plots": ["Train", "Learning", "Test"],
        "plot_now": true,
        "quiet": true,
        "initializer": "xavier"
    }
}
```
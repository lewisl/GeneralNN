
import PyPlot  # avoid namespace confusion with using Plots -- requires qualifying all names from PyPlot


"""

    function save_params(jld2_fname, nnp, bn, hp; train_preds=[], test_preds=[])

Save the trained parameters: nnp, batch_norm parameters: bn, and hyper parameters: hp,
as a JLD2 file.  Note:  be sure to use the jld2 file extension NOT jld as the formats are 
not compatible.

Can be used to run the model on prediction data or to evaluate other
test data results (cost and accuracy).
"""
function save_params(jld_fname, nnp, bn, hp; train_y=[], test_y=[])
    # check if output file exists and ask permission to overwrite
    if isfile(jld_fname)
        print("Output file $jld_fname exists. OK to overwrite? ")
        resp = readline()
        if contains(lowercase(resp), "y")
            rm(jld_fname)
        else
            error("File exists. Replied no to overwrite. Quitting.")
        end
    end

    # to translate to unnormalized regression coefficients: m = mhat / stdx, b = bhat - (m*xmu)

    # write the JLD formatted file (based on hdf5)
    jldopen(jld_fname, "w") do f
        f["nnp"] = nnp
        f["hp"] = hp
        f["bn"] = bn
        if size(train_y) != (0, ) 
              f["train_y"] = train_y 
        end
        if size(test_y) != (0, )
            f["test_y"] = test_y 
        end
    end

end


"""

    function load_params(jld_fname)

Load the trained parameters: nnp, batch_norm parameters: bn, and hyper parameters: hp,
from a JLD file.

Can be used to run the model on prediction data or to evaluate other
test data results (cost and accuracy).

returns: nnp, bn, hp
These are mutable structs.  Use fieldnames(nnp) to list the fields.
"""
function load_params(jld_fname)
    f = jldopen(jld_fname, "r")
    ret = Dict(j=>f[j] for j in keys(f))
    close(f)
    f = []   # flush it before gc gets to it
    return ret
end


function save_plotdef_jld2(plotdef)
    fname = repr(Dates.now())
    fname = "plotdef-" * replace(fname, r"[.:]", "-") * ".jld2"
    jldopen(fname, "w") do f
        f["plotdef"] = plotdef
    end
end


function printby2(nby2)  # not used currently
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end


"""Print the sizes of matrices you pass in as Dict("name"=>var,...)"""
function printdims(indict)
    println("\nSizes of matrices\n")
    for (n, item) in indict
        println(n,": ",size(item))
    end
    println("")
end


"""
Pass a dim1 x dim2 by 1 column vector holding the image data to display it.
Also pass the dimensions as 2 element vector (default is [28,28]).
"""
function display_mnist_digit(digit_data, digit_dims=[28,28])
    PyPlot.ion()
    PyPlot.matshow(reshape(digit_data, digit_dims...)'); # transpose because inputs were transposed
    PyPlot.axis("off")
    println("Press enter to close image window..."); readline()
    PyPlot.close()
end
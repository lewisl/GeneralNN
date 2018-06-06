


"""

    function save_params(jld2_fname, nnp, bn, hp; train_preds=[], test_preds=[])

Save the trained parameters: nnp, batch_norm parameters: bn, and hyper parameters: hp,
as a JLD2 file.  Note:  be sure to use the jld2 file extension NOT jld as the formats are 
not compatible.

Can be used to run the model on prediction data or to evaluate other
test data results (cost and accuracy).
"""
function save_params(jld_fname, nnp, bn, hp; train_preds=[], test_preds=[])
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
    f = jldopen(jld_fname, "w")
    f["nnp"] = nnp
    f["hp"] = hp
    f["bn"] = bn
    if size(train_preds) != (0, ) ;  f["train_y"] = train_preds; end
    if size(test_preds) != (0, ) ;  f["test_y"] = test_preds; end
    
    close(f)

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
    return ret
end



function printby2(nby2)  # not used currently
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end


"""Print the sizes of matrices you pass in as Dict("name"=>var,...)"""
function printdims(indict)
    n = length(indict)
    println("\nSizes of matrices\n")
    for (n, item) in indict
        println(n,": ",size(item))
    end
    println("")
end
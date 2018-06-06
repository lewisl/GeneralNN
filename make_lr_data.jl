using MAT

"""
Create test data for linear regression as a matlab file
"""
function make_lr_data(x_range::Array{Float64,2}, n_examples::Int64,
slope::Array{Float64,1}, y_std::Float64, y_val::Float64;
matfname::String="lrdat.mat",y_type="mean")

    # datafiles follow the convention of rows as examples and columns as features
    # matches most published data, especially for r or python
    # this is slightly slower in Julia and c--train_nnj transposes inputs

    if !in(y_type,["mean", "intercept"])
        error("y_type must be \"mean\" or \"intercept\".")
    end

    # make the x variables
    x = Array{Float64}(n_examples, size(x_range,1))

    for i = 1:size(x_range,1)  # for each row
        x[:, i] = rand(n_examples) .* x_range[i,2] # max of x[,i]
        x[:, i] .= max.(max.(x_range[i,1], x[:,i] .- x_range[i,1]), x[:,i]) # min of x[,i]
    end

    # make y
    y = Array{Float64}(n_examples)
    if y_type == "mean"
        b = y_val - sum((x_range[:,2] .- x_range[:,1]) .* slope)
    elseif y_type == "intercept"
        b = y_val
    end
    
    y[:] =  x * slope .+ b .+ (y_std .* randn(n_examples))

    # write matlab format file
    out = Dict("train" => Dict("x" => x, "y" => y))
    matwrite(matfname, out)

    return out, slope, b
end
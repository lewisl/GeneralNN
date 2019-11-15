

# DONE


# TODO 
#   do we need to do affine when we create the fully connected layer?
#   cut the piece in loops rather than using slicing notation
#   add bias term for convolutional layers
#   do examples loop across layers not per layer
#   implement "no check" versions for convolve and other image functions



using GeneralNN  # must be installed in LOAD_PATH
using Random
using Statistics



function basic(matfname, norm_mode="minmax"; unroll=false, pad=0)
    # create data containers
    train = GeneralNN.Model_data()  # train holds all the data and layer inputs/outputs
    test = GeneralNN.Model_data()
    Random.seed!(1)

    # load training data and test data (if any)
    train.inputs, train.targets, test.inputs, test.targets = GeneralNN.extract_data(matfname)

    # fix the funny thing that MAT file extraction does to the type of the arrays
    train_x = convert(Array{Float64,2}, train.inputs)
    train_y = convert(Array{Float64,2}, train.targets)
    test_x = convert(Array{Float64,2}, test.inputs)
    test_y = convert(Array{Float64,2}, test.targets)

    # set some useful variables
    in_k,n = size(train_x)  # number of features in_k (rows) by no. of examples n (columns)
    out_k = size(train_y,1)  # number of output units
    dotest = size(test_x, 1) > 0  # there is testing data

    norm_factors = GeneralNN.normalize_inputs!(train_x, "minmax")
    dotest && normalize_inputs!(test_x, norm_factors, "minmax") 

    # debug
    # return any(isnan.(train.inputs))



    # structure of convnet
        # layer 1: inputs are 400 x 5000 examples
                    # the image is 20 x 20
        # layer 2: first conv result 18 x 18 by 8 channels = 5408 values
                    # first filters are 3 x 3 by 8 filters = 72 Wgts
                    # first relu is same:   x 5000
        # layer 3: second conv result 16 x 16 by 12 output channels = 6912 values
                    # second filters are 3 x 3 x 8 channels by 12 filters =  864 Wgts
                    # second relu is same by 5000
                    # maxpooling output is 8 x 8 x 12 = 1728 values (no Wgts)
        # layer 4: fc is 240 x 5000 => flatten imgstack to 768 x 5000; affine weight (240, 768))
                    # third relu is 240 by 5000
        # layer 5: affine and softmax => affine weight (10, 2450)
                    # softmax is 10 x 5000

    

    # debug
    # println(join((x_out, y_out, pad), " "))
    # error("that's all folks...")

    ########################################################################
    #  feed forward
    ########################################################################

    # first conv z2 and a2
    # image size values
    filx = fily = 3
    imgx = imgy = Int(sqrt(in_k))
    imgstack = reshape(train_x, imgx, imgy, 1, :)
    a1 = imgstack # effectively, an alias--no allocation
    println("\nsize of image stack: ", size(imgstack))
    stri = 1
    inch = 1  # in channels--channels of input image
    outch = 8 # out channels--new channels for each image filter
    w2 = rand(filx,fily,inch,outch)
    bias2 = fill(0.3,outch)
    (x_out, y_out, pad) = new_img_size(imgstack, filx, fily; stri=stri, pad=pad, same=false)

    z2 = zeros(x_out,y_out,outch,n)  # preallocate for multiple epochs
    if !unroll
        @time for ci = 1:n     #size(imgstack,4)  # ci = current image
            z2[:,:,:, ci] = convolve_multi(imgstack[:,:,:,ci], w2; stri=stri, pad=0)   
        end
    else  # more than 13 times faster than FFT style (stack)!
        println("first unroll and convolve")

        @time unfil = unroll_fil(imgstack[:,:,:,1], w2) # only need to do this once
        @time for ci = 1:n
            z2[:,:,:, ci] = convolve_unroll_all(imgstack[:,:,:,ci], unfil, filx, fily, stri=stri, pad=pad)    
        end
        println("convolved z2 using unroll: ", size(z2))
    end
    # add bias to each out channel
    for i = 1:outch
        z2[:,:,i,:] .+= bias2[i]
    end

    # TODO -- do we want to reuse memory or allocate more?
    # first relu
    println("first relu")
    a2 = copy(z2)
    GeneralNN.relu!(a2, z2)  
    println("type of relu output a2: ", typeof(a2))
    println("size of relu output a2: ", size(a2))

    # second conv z3 and a3
    # image size values
    filx = fily = 3
    imgx, imgy = size(a2)  # from previous conv layer
    stri = 1
    inch = outch # previous out
    outch = 12
    w3 = rand(filx,fily,inch,outch)
    bias3 = fill(0.3, outch)
    (x_out, y_out, pad) = new_img_size(z2, filx, fily; stri=stri, pad=0, same=false)

    z3 = zeros(x_out, y_out, outch, n)
    if !unroll
        for ci = 1:n
            z3[:,:,:, ci] .= convolve_multi(a2[:,:,:, ci], w3; stri=stri)
        end
    else
        println("\nsecond unroll and convolve")
        println("size z3: ", size(z2), " size w3: ", size(w3))
        @time unfil = unroll_fil(z2[:,:,:,1], w3)
        @time for ci = 1:n
            z3[:,:,:, ci] = convolve_unroll_all(a2[:,:,:,ci], unfil, filx, fily, stri=stri, pad=pad)    
        end
    end
    # add bias to each out channel
    @time for c = 1:outch
        z3[:,:,c,:] .+= bias3[c]
    end

    println("type of 2nd conv, layer 3 output: ", typeof(z3))
    println("size of 2nd conv, layer 3 output: ", size(z3))

    # second relu
    println("\nsecond relu")
    @time begin
        a3 = copy(z3)
        GeneralNN.relu!(a3, z3)
    end
    println("type of relu output a3: ", typeof(a3))
    println("size of relu output a3: ", size(a3))

    # maxpooling a3pool
    println("\nmax pooling")
    @time begin
        a3x, a3y, a3c, a3n = size(a3)
        a3pool = zeros(Int(a3x/2), Int(a3y/2), a3c, a3n)  # TODO need reliable way to set pool output size
        a3pool_loc = Array{CartesianIndex{2},4}(undef, Int(a3x/2), Int(a3y/2), a3c, a3n) 
        for i in 1:a3n
            a3pool[:,:,:,i], a3pool_loc[:,:,:,i] = maxpooling(a3[:,:,:,i])
        end
    end
    println("size maxpooling: ", size(a3pool))
    println("size maxpool loc: ", size(a3pool_loc))
    println("type of maxpool loc ", typeof(a3pool_loc))    

    # layer 4: fully connected and relu  z4 and a4
    println("\nfully connected and relu activation z4 and a4")
    in_k = prod(size(a3pool)[1:3])
    out_k = 240
    theta4 = rand(out_k, in_k)
    bias4 = fill(0.3, out_k)
    a4 = z4 = rand(out_k, n) # initialize and allocate
    @time begin
        GeneralNN.affine!(z4, flatten_img(a3pool), theta4, bias4)
        GeneralNN.relu!(a4, z4)
    end
    println("size of fc and relu a4: ", size(a4))

    # layer 5: fully connected and softmax    z5 and a5
    println("\nfully connected z5 and softmax output a5")
    in_k = out_k  # previous layer out
    out_k = 10
    theta5 = rand(out_k, in_k)
    bias5 = fill(0.3, out_k)
    println("size theta4: ", size(theta5))
    a5 = z5 = rand(out_k, n) # initialize and allocate
    @time begin
        GeneralNN.affine!(z5, a4, theta5, bias5)
        GeneralNN.softmax!(a5, z5)
    end
    println("output size after linear and softmax: ", size(a5))

    ########################################################################
    #  back prop
    ########################################################################
        println("\n*****************")
        println("start of backprop")
        # output layer, layer 5--we only need epsilon, the difference
        dz5 = a5 .- train.targets  # called epsilon in FF nn's
        delta_w_5 = dz5 * a4'  # do we need 1/m -- we usually take the average as part of the weight update
        delta_b_5 = sum(dz5, dims=2) # ditto
        println("size of dz5: ", size(dz5))
        println("\nbackprop fully connected layer 5")
        println("size of delta_w_5: ", size(delta_w_5))
        println("size of delta_b_5: ", size(delta_b_5))
        println("size of dz5: ", size(dz5))

        # fully connected, layer 4
        println("\nbackprop fully connected layer 4")
        grad_a4 = zeros(size(a4))
        GeneralNN.relu_gradient!(grad_a4, a4)
        dz4 = theta5' * dz5 .* grad_a4
        delta_w_4 = dz4 * flatten_img(a3pool)'  # this seems weird
        delta_b_4 = sum(dz4, dims=2)

        println("size of delta_w_4: ", size(delta_w_4))
        println("size of delta_b_4: ", size(delta_b_4))
        println("size of dz4: ", size(dz4))

        # maxpooling
            # TODO we need to convert back to an imgstack.  we need to know the dimensions so we need a place
            #    to save dimensions per layer
            # unpooling
            #     max:  need a mask for where the max value is.  deriv is 1.  0 for the other values
            #     avg:  use 1/size of pooling grid times each value.
        println("\nbackprop max pooling")
        pre_unpool3 = theta4' * dz4
        un_pool_3 = zeros(a3x,a3y,a3c,a3n)
        imgstack = reshape(pre_unpool3, Int(a3x/2),Int(a3y/2),a3c,a3n)
        @time for i = 1:a3n
            un_pool_3[:,:,:,i] = unpool(imgstack[:,:,:,i], a3pool_loc[:,:,:,i], mode="max")
        end
        println("size of un_pool3:", size(un_pool_3))


        # 2nd conv
        # un_pool_3 is (16,16,12,5000)   z2 is (18,18,8,5000)
        # w3 is (3,3,12)

        println("\nbackprop of layer 3: convolve and relu")
        println("size a3: ", size(a3), " size un_pool_3: ", size(un_pool_3), " size w3: ", size(w3))
        grad_relu_3 = zeros(size(a3))   # zeros(size(un_pool_3))
        GeneralNN.relu_gradient!(grad_relu_3, a3)
        println("size grad of relu at layer 3: ", size(grad_relu_3))

        dz3 = zeros(size(a3))
        dz3[:] = grad_relu_3 .* un_pool_3
        println("size dz3: $(size(dz3))")

        delta_w_3 = zeros(size(w3))
        @time for i = 1:n 
            delta_w_3[:,:,:,:] += convolve_grad_w(a2[:,:,:,i], dz3[:,:,:,i],  w3)  # middle term? un_pool_3[:,:,:,i],
        end
        delta_w_3[:] = (1/n) .* delta_w_3
        delta_b_3 = (1/n) .* sum(dz3 ,dims=(1,2,4))[:]   # alternative to sum? un_pool_3
        println("size delta_w_3: $(size(delta_w_3)) size delta_b_3: $(size(delta_b_3))")

        # 1st relu at layer 2
        println("\nbackprop of layer 2: convolve and relu")
        println("size a2: ", size(a2),  " size w2: ", size(w2))
        grad_relu_2 = zeros(size(a2))   # zeros(size(un_pool_3))
        GeneralNN.relu_gradient!(grad_relu_2, a2)
        println("size grad of relu at layer 2: ", size(grad_relu_2))

        dz2 = zeros(size(a2))
        @time for i = 1:n
            dz2[:,:,:, i] = convolve_grad_x(dopad(dz3[:,:,:,i],2), w3)  # alternative to pad? un_pool_3
        end
        println("initial size of dz3 before gradient: $(size(dz2))")
        dz2[:] = dz2 .* grad_relu_2
        println("size dz2: $(size(dz2))")

        delta_w_2 = zeros(size(w2))
        @time for i = 1:n 
            delta_w_2[:,:,:,:] += convolve_grad_w(a1[:,:,:,i], dz2[:,:,:,i], w2)
        end
        delta_w_2[:] = (1/n) .* delta_w_2
        delta_b_2 = (1/n) .* sum(dz2,dims=(1,2,4))[:]
        println("size delta_w_2: $(size(delta_w_2)) size delta_b_2: $(size(delta_b_2))")
            
    
    println("that's all folks!...")

end


"""
Convolve a one or multi-channel image with a filter with one or more output channels.
This is a 20x speedup over array broadcasting.
"""
function convolve_multi(img, fil; same=false, stri=1, mode="normal")   # took out this arg: , pad=0
    # TODO try another version use elementwise multiplication on views and sum the result
    # this way is faster! 
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    if ndims(fil) == 2
        filx, fily = size(fil)
        filc = filp = 1
        fil = reshape(fil,filx, fily, filc, filp)
    elseif ndims(fil) == 3  # one filter
        filx, fily, filc = size(fil)
        filp = 1
        fil = reshape(fil,filx, fily, filc, filp)       
    elseif ndims(fil) == 4  # multiple filters
        filx, fily, filc, filp = size(fil)  # filc = filter channels must equal image channels; filp = filter planes--number of output channels
    else
        error("wrong number of dimensions for filter: $(ndims(fil))")
    end

    if !(filc == imgc)                    # & (mode == "normal")
        error("Number of channels in image and filter do not match.")
    end

    # if same 
    #     pad = ceil(Int, (filx - 1) / 2)
    # end

    # if pad > 0
    #     img = dopad(img, pad)
    # end

    # dimensions of the single plane convolution result
    x_out = floor(Int, (imgx - filx) / stri) + 1
    y_out = floor(Int, (imgy - fily) / stri) + 1

    ret = zeros(x_out, y_out, filp)
    for z = 1:filp  # new channels
        for j = zip(1:y_out, 1:stri:imgy)  # column major access
            for i = zip(1:x_out, 1:stri:imgx) # 1st steps through ret; 2nd steps through image subset
                element = 0.0
                for ic = 1:imgc, fj = 1:fily, fi = 1:filx  # input image channels  # scalar multiply faster than slice & broadcast
                    element += img[i[2]+fi-1,j[2]+fj-1,ic] * fil[fi, fj, ic, z]
                end
                ret[i[1],j[1],z] = element
            end
        end
    end

    return ret
end


function convolve_grad_w(x, dz, w)
    # println("size x: $(size(x)) size dz: $(size(dz)) size w: $(size(w))")
    ret = zeros(size(w))
    inch = size(x,3)
    outch = size(dz,3)
    for g = 1:outch # this loop does a bunch of 1 plane convolutions and packs them into the w_grad array
       for c = 1:inch
           ret[:,:,c,g] = convolve_multi(x[:,:,c],dz[:,:,g])   # with scalar for index 3, these are 2D convolutions
       end
    end
    # println("size of grad_w for one sample: $(size(ret))")
    return ret
end


function convolve_grad_x(x,w)
    # println("size x: $(size(x)) size w: $(size(w))")
    imgx,imgy,imgc = size(x)
    filx, fily, outch, inch = size(w) # note for grad we reverse the number of in and out channels
    imgx -= filx-1 
    imgy -= fily-1
    ret = zeros(imgx,imgy,outch)
    for g = 1:outch
        ret[:,:,g] = convolve_multi(x, w[:,:,g,:])
    end
    return ret
end


function new_img_size(img, filx, fily; pad = 0, stri = 1, same=false)
    imgx, imgy = size(img)
    new_img_size(imgx, imgy, filx, fily; pad=pad, stri=stri, same=same)
end


function new_img_size(imgx, imgy, filx, fily; pad = 0, stri = 1, same=false)
    # dimensions of the single plane convolution result
    x_out = floor(Int, (imgx + 2 * pad - filx ) / stri) + 1
    y_out = floor(Int, (imgy + 2 * pad - fily ) / stri) + 1
    return(x_out, y_out, pad)
end

"""
    dopad(arr, pad; padval=0)
    a complicated one-liner:  yuck--but, it's 15 times faster that vector catenating!

    arr is a 2d image, optionally with channels.
    pad an integer for the number of border elements to pad.
    padval is the value to pad with.  Default 0 is nearly always the one you'll use.

    Returns padded array as: arr[m,n,c] with c = 1 or number of input channels.

    Alternative method for imgstack: dopad(arr, pad, 4; padval=0)
    You must supply the dims argument as 4 to signal you have a tensor imgstack: arr[x,y,c,z].

    Returns padded array as: arr[m,n,c,z] with z = 1 or number of input channels.
"""
function dopad(arr,pad; padval=0) 
    padval = convert(eltype(arr), padval)
    m,n = size(arr)
    c = ndims(arr) == 3 ? size(arr,3) : 1
    return [(i in 1:pad) || (j in 1:pad) || (i in m+pad+1:m+2*pad) || (j in n+pad+1:n+2*pad) ? padval : 
        arr[i-pad,j-pad, z] for i=1:m+2*pad, j=1:n+2*pad, z=1:c]
end


function dopad(arr, pad, dims; padval=0) 
    dims != 4 && error("dims argument value must be 4 for array as 4 dimensional tensor")
    padval = convert(eltype(arr), padval)
    m,n = size(arr)
    c = size(arr,3)
    k = size(arr,4)
    return [(i in 1:pad) || (j in 1:pad) || (i in m+pad+1:m+2*pad) || (j in n+pad+1:n+2*pad) ? padval : 
        arr[i-pad,j-pad, z, cnt] for i=1:m+2*pad, j=1:n+2*pad, z=1:c, cnt=1:k]
end


function flip(arr)
    m,n = (size(arr,1), size(arr,2))
    if !(m == n)
        error("Input filter array must be square.")
    end
    T = eltype(arr)
    ret = zeros(T,m,n)
    st = ceil(Int,m/2)
    for i = 1:m 
        for j = 1:st
            d = arr[i, j]
            new_i = m - (i - 1)
            new_j = m - (j - 1)

            ret[i, j] = arr[new_i, new_j]
            ret[new_i, new_j] = d
        end
    end
    return ret
end


function pooling(img; pooldims=[2,2], same=false, stri=2, pad=0, mode="max")
    if mode=="max"
        pfunc = maximum  # pfunc => pool function
    elseif mode=="avg"
        pfunc = mean
    else
        error("mode must be max or avg")
    end

    img_x,img_y = size(img,1), size(img,2)
    c = ndims(img) == 3 ? size(img, 3) : 1

    poolx,pooly = pooldims

    # if same 
    #     pad = ceil(Int, (poolx - 1) / 2)
    # end

    # if pad > 0
    #     img = dopad(img, pad)
    # end

    # dimensions of the single plane pooling result
    x_out = floor(Int, (img_x + 2 * pad - poolx ) / stri) + 1
    y_out = floor(Int, (img_y + 2 * pad - pooly ) / stri) + 1

    ret = zeros(x_out, y_out, c)
    loc = Array{CartesianIndex{2},3}(undef,x_out, y_out,c)
    # loc = fill(falses(poolx, pooly),(x_out, y_out, c))
    for z = 1:c
        for i = zip(1:x_out, 1:stri:img_x)
            for j = zip(1:y_out, 1:stri:img_y)
                submatview = @view(img[i[2]:i[2]+poolx-1, j[2]:j[2]+pooly-1, z])  # view saves 15 to 20 percent
                val = pfunc(submatview)  
                ret[i[1], j[1], z] = val
                # loc[i[1], j[1], z] = isapprox.(val, submatview)
                mode == "max" && (loc[i[1], j[1], z] = findfirst(x -> isapprox(x, val),submatview))
            end
        end
    end

    return ret, loc
end

function unpool(dx, dloc; pooldims = (2,2), mode="max") # works per image not across entire imgstack
    m, n, c = size(dx)  # fix to handle channels
    poolx, pooly = pooldims
    ret = zeros(m*poolx, n*pooly, c)
    if mode == "max"
        for z = 1:c
            for i = 1:m 
                for j = 1:n 
                    # ret[(i-1)*poolx+1:i*poolx, (j-1)*pooly+1:j*pooly] = fill(dx[i,j], poolx, pooly) .* dloc[i,j]
                    subret = @view(ret[(i-1)*poolx+1:i*poolx, (j-1)*pooly+1:j*pooly, z])
                    subret[dloc[i,j,z]] = dx[i,j,z]
                    # println(fill(dx[i,j], poolx, pooly))
                end
            end
        end
    elseif mode == "avg"
        scale = 1 / sum(pooldims)
        for z = 1:c
            for i = 1:m 
                for j = 1:n 
                    # ret[(i-1)*poolx+1:i*poolx, (j-1)*pooly+1:j*pooly] = fill(dx[i,j], poolx, pooly) .* dloc[i,j]
                    subret = @view(ret[(i-1)*poolx+1:i*poolx, (j-1)*pooly+1:j*pooly, z])
                    fillit = scale * dx[i,j,z]
                    fill!(subret, fillit)
                    # println(fill(dx[i,j], poolx, pooly))
                end
            end
        end
    else
        error("mode must be max or avg")
    end    
    return ret
end


function avgpooling(img; pooldims=[2,2], same=false, stri=2, pad=0)
    pooling(img; pooldims=pooldims, same=same, stri=stri, pad=pad, mode="avg")
end


function maxpooling(img; pooldims=[2,2], same=false, stri=2, pad=0)
    pooling(img; pooldims=pooldims, same=same, stri=stri, pad=pad, mode="max")
end


"""
function flatten_img(img)

    flatten a 2d or 3d image for a conv net to use as a fully connected layer.
    follow the convention that features are rows and examples are columns.
    each column of an image plane is stacked below the prior column.  the next
    plan (or channel) comes below the prior channel.
    The size of img must have 4 dimensions even if each image has only 1 channel.
    For example:  7 6x6 images each with 1 channel should have dims 6,6,1,7.

    Note that this returns an array that is a view of the input array.
"""
function flatten_img(imgstack)
    if ndims(imgstack) != 4
        error("imgstack must have 4 dimensions even if img is 2d (use 1 for number of channels)")
    end
    x,y,c,n = size(imgstack)  # m x n image with c channels => z images like this
    return reshape(imgstack, prod((x,y,c)), n)
end


"""
function stack_img(fc, imgdims)

    Convert a flattened image back to an image stack.
    imgdims must provide 3 integer values:  m (rows or height) x n (columns or width) x c (number of channels).
    c, number of channels, must be provided for 2D images:  just use 1.

    returns: an imagestack that is imgx x imgy x nchannels x k where k is the number of images (or examples)

    Note: this returns an array that is a view of the input array.
"""
function stack_img(fc, imgdims)
    if length(imgdims) != 3
        error("imgdims must contain 3 integer values")
    end

    if ndims(fc) != 2
        error("fc must a 2d array with rows for image data and a column for each image")
    end

    imgx, imgy, nchannels = imgdims
    fcx, k = size(fc)

    if fcx != prod(imgdims)
        error("number of rows--number of elements for each image--does not match the image dimensions")
    end

    return reshape(fc, imgx, imgy, nchannels, k)
end

function stack_img(fc, xdim::Int, ydim::Int, nchannels::Int)
    if ndims(fc) != 2
        error("fc must a 2d array with rows for image data and a column for each image")
    end

    fcx, k = size(fc)

    if fcx != prod((xdim, ydim, nchannels))
        error("number of rows--number of elements for each image--does not match the image dimensions")
    end

    return reshape(fc, xdim, ydim, nchannels, k)

end



"""
make a 3d array by copying the 2d array into the 3rd dimension
"""
function catn(array, n::Integer)
    if n < 1
        error("n must be an integer greater than equal to 1")
    end
    ret = array
    for i = 1:n-1
        ret = cat(ret, array, dims = 3)
    end
    return ret
end

####################################################################
#  unrolling approach from a paper by MS Research
#     allows matrix multiplication to convolve
#     crazy fast compared to 2D geom approach
####################################################################
#   should we unroll the whole imgstack or just go one at a time?  depends on whether we can reuse
#   implement stride, padding, same for unrolled convolutions

function convolve_unroll_all(img, unfil, filx, fily; stri=1, pad=0, same=false)
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    (x_out, y_out, pad) = new_img_size(img, filx, fily; stri=stri, pad=pad, same=false)
    unimg = zeros(x_out+2*pad, (y_out+2*pad)*filx*fily, imgc)

    unimg[:] = unroll_img(img, filx, fily; stri=stri, pad=pad, same=same)
    return convolve_unroll(unimg, unfil)
end


function unroll_img(img, fil; stri=1, pad=0, same=false)
# method using img rather than dims of image

    filx, fily = size(fil)
    unroll_img(img, filx, fily; stri=stri, pad=pad, same=same)
end


function unroll_img(img, filx, fily; stri=1, pad=0, same=false)
# method using the x and y dimensions of the filter
# TODO remove pad argument

    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    # if pad > 0
    #     img = dopad(img,pad)
    # end

    l_fil = filx * fily
    x_out, y_out, pad = new_img_size(img, filx, fily; stri=stri, pad=pad)
    unimg = Array{eltype(img),3}(undef,x_out,y_out*filx*fily, imgc)

    # debug
    # println("unimg ",size(unimg))
    # println("img   ", size(img))

    for z = 1:imgc
        for i = 1:x_out 
            for j = 1:y_out 
                t = 0
                for m=i:i+filx-(max(2*pad+1,1))
                    for n=j:j+fily-(max(2*pad+1,1)) 
                        t += 1  # column displacement (across part of row) for result matrix
                        unimg[i,(j-1)*l_fil+t, z] = img[m,n, z]  
                        # println(m," ", n, " ",x[m,n])
                    end
                end
            end
        end
    end

    return unimg
end


function unroll_fil(img, fil; stri=1, pad=0, same=false)
# method with input for img
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    unroll_fil(imgx, imgy, imgc, fil; stri=stri, pad=pad, same=same)
end


function unroll_fil(imgx, imgy, imgc, fil; stri=1, pad=0, same=false)
# method with inputs for imgx, imgy, imgc

    # filc = filter channels must equal image channels; filp = filter planes--number of output channels
    if ndims(fil) == 2 # one filter, one image channel
        filx, fily = size(fil)
        filc = filp = 1
    elseif ndims(fil) == 3  # one filter with multiple image channels
        filx, fily, filc = size(fil)
        filp = 1
    elseif ndims(fil) == 4  # multiple filters and multiple image channels
        filx, fily, filc, filp = size(fil)  
    else
        error("wrong number of dimensions for filter: $(ndims(fil))")
    end

    !(filc == imgc) && error("Number of channels in image and filter do not match.")   

    l_fil = filx * fily
    x_out, y_out, pad = new_img_size(imgx, imgy, filx, fily; stri=1, pad=pad)
    fil = reshape(fil,filx,fily, filc, filp)      # TODO this will change the sender OK????

    flat = reshape(permutedims(fil,[2,1,3,4]), l_fil, filc, filp)
    unfil = zeros(eltype(fil), (y_out*filx*fily, x_out, filc, filp))  

    # debug
    # println("flat  ", size(flat))
    # println("unfil ", size(unfil))

    # for z = 1:filp
        for j = 1:x_out
            st = (j-1) * l_fil + 1
            fin = st + l_fil - 1
            unfil[st:fin,j,:,:] = flat
        end
    # end
    return unfil
end



function convolve_unroll(img, fil)
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    if ndims(fil) == 2 # one filter, one image channel
        filx, fily = size(fil)
        filc = filp = 1
    elseif ndims(fil) == 3  # one filter with multiple image channels
        filx, fily, filc = size(fil)
        filp = 1
    elseif ndims(fil) == 4  # multiple filters and multiple image channels
        filx, fily, filc, filp = size(fil)  
    else
        error("wrong number of dimensions for filter: $(ndims(fil))")
    end

    if !(filc == imgc) 
        error("Number of channels in image and filter do not match.")
    end    

    ret = zeros(imgx, fily, filp)
    for z = 1:filp
        for c = 1:imgc
            ret[:,:,z] .+= @view(img[:,:,c]) * @view(fil[:,:,c,z])
        end
    end

    return ret
end


####################################################################
#  performance and other experiments that didn't make it
####################################################################

# this one is 3X slower!
function convolve_multi2(img, fil; same=false, stri=1)   # took out this arg: , pad=0
    # TODO try another version use elementwise multiplication on views and sum the result
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    if ndims(fil) == 3  # one filter
        filx, fily, filc = size(fil)
        filp = 1
    elseif ndims(fil) == 4  # multiple filters
        filx, fily, filc, filp = size(fil)  # filc = filter channels must equal image channels; filp = filter planes--number of output channels
    else
        error("wrong number of dimensions for filter: $(ndims(fil))")
    end

    if !(filc == imgc)   
        error("Number of channels in image and filter do not match.")
    end

    # if same 
    #     pad = ceil(Int, (filx - 1) / 2)
    # end

    # if pad > 0
    #     img = dopad(img, pad)
    # end

    # dimensions of the single plane convolution result
    x_out = floor(Int, (imgx - filx) / stri) + 1
    y_out = floor(Int, (imgy - fily) / stri) + 1

    ret = zeros(x_out, y_out, filp)
    for z = 1:filp
        for j = zip(1:y_out, 1:stri:imgy)  # column major access
            for i = zip(1:x_out, 1:stri:imgx)
                ret[i[1],j[1],z] = sum((@view img[i[2]:i[2]+filx-1,j[2]:j[2]+fily-1,:]) .* (@view fil[:,:,:,z]))
            end
        end
    end

    return ret
end


####################################################################
# data and filters to play with
####################################################################

x = [3 0 1 2 7 4; 
     1 5 8 9 3 1;
     2 7 2 5 1 3;
     0 1 3 1 7 8;
     4 2 1 6 2 8;
     2 4 5 2 3 9]

x3d = cat(x,x,x, dims = 3)

v_edge_fil = [1 0 -1;
              1 0 -1;
              1 0 -1]

# no need to copy the same filter--3D filter only makes sense with each plane being a different filter
v_edge_fil3d = cat(v_edge_fil,v_edge_fil,v_edge_fil, dims = 3)

h_edge_fil = [1 1 1;
              0 0 0;
              -1 -1 -1]

sobel_fil =  [1 0 -1;
              2 0 -2;
              1 0 -1]

schorr_fil =  [3 0 -3;
              10 0 -10;
              3 0 -3]

edg = [10 10 10 0 0 0; 
     10 10 10 0 0 0;
     10 10 10 0 0 0;
     10 10 10 0 0 0;
     10 10 10 0 0 0;
     10 10 10 0 0 0]

edg3d = cat(edg, edg, edg, dims = 3)

edg2 =  [10 10 10 0 0 0; 
         10 10 10 0 0 0;
         10 10 10 0 0 0;
         0 0 0 10 10 10;
         0 0 0 10 10 10;
         0 0 0 10 10 10]


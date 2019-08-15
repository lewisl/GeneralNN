

# DONE


# TODO 
#   cut the piece in loops rather than using slicing notation
#   add bias term for convolutional layers
#   do examples loop across layers not per layer
#   implement "no check" versions for convolve and other image functions



using GeneralNN  # must be installed in LOAD_PATH
using Random



function basic(matfname, norm_mode="minmax", unroll=false)
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
        # inputs are 784 x 5000 examples
        # the image is 28 x 28
        # first conv 26 x 26 by 8 channels = 5408 values
        #     first filters are 3 x 3 by 8 = 72 weights
        # first relu is same:  5408 x 5000
        # second conv 24 x 24 by 12 output channels = 6912 values
        #     second filters are 3 x 3 x 8 x 12 =  864 weights
        # second relu is same 6912 by 5000
        # maxpooling output is 12 x 12 x 12 = 1728 values (no weights)
        # fc is 1728 x 5000  (reshape)
        # softmax is 10 x 5000

    

    # debug
    # println(join((x_out, y_out, pad), " "))
    # error("that's all folks...")

    ########################################################################
    #  feed forward
    ########################################################################

    # first conv
    # image size values
    filx = fily = 3
    imgx = imgy = Int(sqrt(in_k))
    imgstack = reshape(train_x, imgx, imgy, 1, :)
    println("size of image stack: ", size(imgstack))
    stri = 1
    inch = 1
    outch = 8
    w1 = rand(filx,fily,inch,outch)
    (x_out, y_out, pad) = new_img_size(imgx, imgy, filx, fily; stri=stri, pad=0, same=false)

    cv1 = zeros(x_out,y_out,8,n)  # preallocate for multiple epochs
    println("size after 1st convolution: ", size(cv1))
    if !unroll
        @time for ci = 1:n     #size(imgstack,4)  # ci = current image
            cv1[:,:,:, ci] = convolve_multi(imgstack[:,:,:,ci], w1; stri=stri)   # TODO did the dot do anything
        end
    elseif unroll  # more than 13 times faster than FFT style (stack)!
        println("first unroll and convolve")
        # unroll all examples in one go
        unimg = zeros(x_out, x_out*filx*fily, inch, n)
        @time for ci = 1:n
            unimg[:,:,:,ci] = unroll_img(imgstack[:,:,:,ci], w1)
        end

        @time unfil = unroll_fil(imgstack[:,:,:,1], w1)
        @time for ci = 1:n
            cv1[:,:,:, ci] = convolve_unroll(unimg[:,:,:,ci], unfil)    
        end

    else
        error("value of unroll not set to true or false.")
    end

    # TODO -- do we want to reuse memory or allocate more?
    # first relu
    println("first relu")
    @time begin
        rl1 = copy(flatten_img(cv1))
        GeneralNN.relu!(rl1, rl1)
        cv1 = stack_img(rl1,(x_out, y_out, outch))  # TODO do we need the pre-relu output of convolve?
    end
    println("type of conv output: ", typeof(cv1))


    # # second conv
    # image size values
    filx = fily = 3
    imgx, imgy = x_out, y_out  # from previous conv layer
    stri = 1
    inch = outch
    outch = 12
    w2 = rand(filx,fily,inch,outch)
    (x_out, y_out, pad) = new_img_size(imgx, imgy, filx, fily; stri=stri, pad=0, same=false)
    cv2 = zeros(x_out, y_out, outch, n)
    if !unroll
        for ci = 1:n
            cv2[:,:,:, ci] .= convolve_multi(cv1[:,:,:, ci], w2; stri=stri)
        end
    else
        # unroll all examples in one go
        println("second unroll and convolve")
        unimg = zeros(x_out, x_out*filx*fily, inch, n)
        @time for ci = 1:n
            unimg[:,:,:,ci] = unroll_img(cv1[:,:,:,ci], w2)
        end

        @time unfil = unroll_fil(cv1[:,:,:,1], w2)
        @time for ci = 1:n
            cv2[:,:,:, ci] = convolve_unroll(unimg[:,:,:,ci], unfil)    
        end
    
    end
    println("type of conv output: ", typeof(cv2))

    # second relu
    println("second relu")
    @time begin
        rl2 = copy(flatten_img(cv2))
        GeneralNN.relu!(rl2, rl2)
        cv2 = stack_img(rl2, (x_out, y_out, outch))  # TODO do we need the pre-relu output of convolve?
    end


    # maxpooling
    println("max pooling")
    @time begin
        cv2x, cv2y, cv2c, cv2k = size(cv2)
        cv2pool = zeros(Int(cv2x/2), Int(cv2y/2), cv2c, cv2k)
        for i in 1:n
            cv2pool[:,:,:,i] = maxpooling(cv2[:,:,:,i])
        end
    end
    println("final size: ", size(cv2pool))

    # fully connected
    println("fully connected and softmax output")
    out_k = 10
    in_k = prod(size(cv2pool)[1:3])
    theta = rand(out_k, in_k)
    bias = rand(out_k)
    a = z = rand(out_k, n)
    @time begin
        GeneralNN.affine!(z, copy(flatten_img(cv2pool)), theta, bias)
        GeneralNN.softmax!(a, z)
    end
    println("output size: ", size(a))

    ########################################################################
    #  back prop
    ########################################################################


    return a
    println("that's all folks!...")

end


"""
Convolve a one or multi-channel image with a filter with one or more output channels.
This is a 20x speedup over array broadcasting.
"""
function convolve_multi(img, fil; same=false, stri=1, pad=0)
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

    if same 
        pad = ceil(Int, (filx - 1) / 2)
    end

    if pad > 0
        img = dopad(img, pad)
    end

    # dimensions of the single plane convolution result
    x_out = floor(Int, (imgx + 2 * pad - filx ) / stri) + 1
    y_out = floor(Int, (imgy + 2 * pad - fily ) / stri) + 1

    ret = zeros(x_out, y_out, filp)
    for z = 1:filp
        for j = zip(1:y_out, 1:stri:imgy)  # column major access
            for i = zip(1:x_out, 1:stri:imgx)
                element = 0.0
                for ic = 1:imgc, fj = 1:fily, fi = 1:filx  # scalar multiply faster than slice and broadcast
                    element += img[i[2]+fi-1,j[2]+fj-1,ic] * fil[fi, fj, ic, z]
                end
                ret[i[1],j[1],z] = element
            end
        end
    end

    return ret
end


function new_img_size(imgx, imgy, filx, fily; pad = 0, stri = 1, same=false)
    if same 
        pad = ceil(Int, (filx - 1) / 2)
    end

    if pad > 0
        img = dopad(img, pad)
    end

    # dimensions of the single plane convolution result
    x_out = floor(Int, (imgx + 2 * pad - filx ) / stri) + 1
    y_out = floor(Int, (imgy + 2 * pad - fily ) / stri) + 1

    return(x_out, y_out, pad)
end


"""
a complicated one-liner:  yuck--but, it's 15 times faster that vector catenating!
"""
function dopad(arr,pad)  # use array comprehension
    m,n = (size(arr,1), size(arr,2))
    c = ndims(arr) == 3 ? size(arr,3) : 1
    return [(i in 1:pad) || (j in 1:pad) || (i in m+pad+1:m+2*pad) || (j in n+pad+1:n+2*pad) ? 0 : 
        arr[i-pad,j-pad, z] for i=1:m+2*pad, j=1:n+2*pad, z=1:c]
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

    if same 
        pad = ceil(Int, (poolx - 1) / 2)
    end

    if pad > 0
        img[:] = dopad(img, pad)
    end

    # dimensions of the single plane convolution result
    x_out = floor(Int, (img_x + 2 * pad - poolx ) / stri) + 1
    y_out = floor(Int, (img_y + 2 * pad - pooly ) / stri) + 1

    ret = zeros(x_out, y_out, c)
    for z = 1:c
        for i = zip(1:x_out, 1:stri:img_x)
            for j = zip(1:y_out, 1:stri:img_y)
                ret[i[1],j[1], z] = pfunc(@view(img[i[2]:i[2]+poolx-1, j[2]:j[2]+pooly-1, z]))  
                    # view saves 15 to 20 percent
            end
        end
    end

    return ret
end


function avgpooling(img; pooldims=[2,2], same=false, stri=2, pad=0, mode="avg")
    pooling(img; pooldims=pooldims, same=same, stri=stri, pad=pad, mode="avg")
end


function maxpooling(img; pooldims=[2,2], same=false, stri=2, pad=0, mode="max")
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
    m,n,c,z = size(imgstack)  # m x n image with c channels => z images like this
    return reshape(imgstack, prod((m,n,c)), z)
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

function unroll_img(img, fil; stri=1, pad=0, same=false)
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    filx, fily = size(fil)[1:2]
    l_fil = filx * fily
    x_out, y_out, pad = new_img_size(imgx, imgy, filx, fily; stri=1)
    unimg = Array{eltype(img),3}(undef,x_out,y_out*filx*fily, imgc)

    # debug
    # println("unimg ",size(unimg))
    # println("img   ", size(img))

    for z = 1:imgc
        for i = 1:x_out 
            for j = 1:y_out 
                t = 0
                for m=i:i+filx-1 
                    for n=j:j+fily-1 
                        t += 1  # column displacement (across part of row) for result matrix
                        unimg[i,(j-1)*l_fil+t, z] = img[m, n, z]  
                        # println(m," ", n, " ",x[m,n])
                    end
                end
            end
        end
    end

    return unimg
end

# method using the x and y dimensions of the filter
function unroll_img(img, filx, fily; stri=1, pad=0, same=false)
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    l_fil = filx * fily
    x_out, y_out, pad = new_img_size(imgx, imgy, filx, fily; stri=1)
    unimg = Array{eltype(img),3}(undef,x_out,y_out*filx*fily, imgc)

    # debug
    # println("unimg ",size(unimg))
    # println("img   ", size(img))

    for z = 1:imgc
        for i = 1:x_out 
            for j = 1:y_out 
                t = 0
                for m=i:i+filx-1 
                    for n=j:j+fily-1 
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
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

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

    if !(filc == imgc) 
        # println("filc ", filc, " imgc ", imgc)
        error("Number of channels in image and filter do not match.")
    end    

    l_fil = filx * fily
    x_out, y_out, pad = new_img_size(imgx, imgy, filx, fily; stri=1)
    fil = reshape(fil,filx,fily, filc, filp)                          # TODO this will change the sender OK????

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

# method with inputs for imgx, imgy, imgc
function unroll_fil(imgx, imgy, imgc, fil; stri=1, pad=0, same=false)

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

    if !(filc == imgc) 
        # println("filc ", filc, " imgc ", imgc)
        error("Number of channels in image and filter do not match.")
    end    

    l_fil = filx * fily
    x_out, y_out, pad = new_img_size(imgx, imgy, filx, fily; stri=1)
    fil = reshape(fil,filx,fily, filc, filp)                          # TODO this will change the sender OK????

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



# data and filters to play with
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


# function convnet()
#     # some useful dimensions

#     # topology of multiple images
#     m,n,c,k  # m rows, n columns, c channels, k images

#     # net architecture by layer
#     arch = Dict(
#                 1 => Dict(
#                     "type" => "conv", # one of conv, max, avg, fc, classify, relu, l_relu, sigmoid, softmax, none
#                     "dims" => [24,24,3]
#                 ),
#                 2 => Dict(
#                     "type" => "conv", # one of conv, max, avg, fc, classify
#                     "unit" => "relu", # one of relu, l_relu, sigmoid, softmax
#                     "dims" => [24,24,3]
#                 ),
#                 3 => Dict(
#                     "type" => "conv", # one of conv, max, avg, fc, classify
#                     "unit" => "relu", # one of relu, l_relu, sigmoid, softmax
#                     "dims" => [24,24,3]
#                 ),
#                 4 => Dict(
#                     "type" => "conv", # one of conv, max, avg, fc, classify
#                     "unit" => "relu", # one of relu, l_relu, sigmoid, softmax
#                     "dims" => [24,24,3]
#                 ),
#     )

#     # number  layers
#     num_layers = maximum(keys(arch))

#     # layer functions -- canonical set
#     func_def = Dict(
#         "conv" = convolve,
#         "relu" = relu,
#         "l_relu" = l_relu,
#         "sigmoid" = sigmoid,
#         "softmax" = softmax,
#         "max" = maxpooling,
#         "avg" = avgpooling,
#         "fc" = fc,
#         "classify" = classify,
#         "none" = donothing,
#     )

#     # set layer functions to use per layer of network architecture
#     layer_funcs = [func_def[arch[i]["type"]] for i in 1:num_layers]

# end


"""
convolve a single plane filter with an image of any depth.
return a single plane output for that filter.
"""
function convolve(img, fil, same=false, stri=1, pad=0)
    imgx, imgy = size(img,1,2)
    filx, fily = size(fil,1,2)

    if same 
        pad = ceil(Int, (filx - 1) / 2)
    end

    if pad > 0
        img = dopad(img, pad)
    end

    # dimensions of the single plane convolution result
    x_out = floor(Int, (imgx + 2 * pad - filx ) / stri) + 1
    y_out = floor(Int, (imgy + 2 * pad - fily ) / stri) + 1

    ret = Array{Float64}(x_out, y_out)
    for i = zip(1:x_out, 1:stri:imgx)
        for j = zip(1:y_out, 1:stri:imgy)
            ret[i[1],j[1]] = sum(img[i[2]:i[2]+filx-1, j[2]:j[2]+fily-1, :] .* fil)  
        end
    end

    return ret
end


"""
a complicated one-liner:  yuck--but, it's 15 times faster that catenating!
"""
function dopad(arr,pad)  # use array comprehension
    m,n = size(arr,1,2)
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

    m,n = size(img,1,2)
    c = ndims(img) == 3 ? size(img,3) : 1

    poolx,pooly = pooldims

    if same 
        pad = ceil(Int, (poolx - 1) / 2)
    end

    if pad > 0
        img = dopad(img, pad)
    end

    # dimensions of the single plane convolution result
    x_out = floor(Int, (m + 2 * pad - poolx ) / stri) + 1
    y_out = floor(Int, (n + 2 * pad - pooly ) / stri) + 1

    ret = Array{Float64}(x_out, y_out, c)
    for z = 1:c
        for i = zip(1:x_out, 1:stri:m)
            for j = zip(1:y_out, 1:stri:n)
                ret[i[1],j[1], z] = pfunc(img[i[2]:i[2]+poolx-1, j[2]:j[2]+pooly-1, z])  
            end
        end
    end

    return ret
end


function avgpooling(img; pooldims=[2,2], same=false, stri=2, pad=0, mode="avg")
    pooling(img; pooldims=pooldims, same=same, stri=stri, pad=pad, mode="avg")
end


"""
function fc(img)

    flatten a 2d or 3d image for a conv net to use as a fully connected layer.
    follow the convention that features are rows and examples are columns.
    each column of an image plane is stacked below the prior column.  the next
    plan (or channel) comes below the prior channel.
    The size of img must have 4 dimensions even if each image has only 1 channel.
    For example:  7 6x6 images each with 1 channel should have dims 6,6,1,7.

    Note that this returns an array that is a view of the input array.
"""
function fc(imgstack)
    if ndims(imgstack) != 4
        error("imgstack must have 4 dimensions even if img is 2d (use 1 for number of channels)")
    end
    m,n,c,z = size(imgstack)  # m x n image with c channels => z images like this
    return reshape(imgstack,prod([m,n,c]),z)
end


"""
function fc_to_stack(fc, imgdims)

    Convert a flattened image stack back to an image stack.
    imgdims must provide 3 integer values:  m (rows or height) x n (columns or width) x c (number of channels).
    c, number of channels, must be provided for 2D images:  just use 1.

    returns: an imagestack that is m x n x c x z where z is the number of images (or examples)
"""
function fc_to_stack(fc, imgdims::Array{Int,1})
    if length(imgdims) != 3
        error("imgdims must contain 3 integer values")
    end

    if ndims(fc) != 2
        error("fc must a 2d array with rows for image data and a column for each image")
    end

    m,n,c = imgdims
    fcx, z = size(fc)

    if fcx != prod(imgdims)
        error("number of rows--number of elements for each image--does not match the image dimensions")
    end

    ret = Array{Float64,4}(m,n,c,z)
    for l = 1:z
        for k = 1:c
            for j = 1:n
                for i = 1:m
                    ret[i,j,k,l] = fc[i + ((j-1)*m) + (k-1)*(m*n) ,l]
                end
            end
        end
    end

    return ret
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
        ret = cat(3, ret,array)
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

x3d = cat(3,x,x,x)

v_edge_fil = [1 0 -1;
       1 0 -1;
       1 0 -1]

# no need to copy the same filter--3D filter only makes sense with each plane being a different filter
v_edge_fil3d = cat(3, v_edge_fil,v_edge_fil,v_edge_fil)

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

edg3d = cat(3, edg, edg, edg)

edg2 =  [10 10 10 0 0 0; 
         10 10 10 0 0 0;
         10 10 10 0 0 0;
         0 0 0 10 10 10;
         0 0 0 10 10 10;
         0 0 0 10 10 10]

# a hard way to pad that is a bit hard to follow
# if pad > 0  # make a padded image
#     p_img = Array{eltype(img),2}(2.+size(img))
#     p_img[2:2+size(img,1)-1,2:2+size(img,2)-1] = img
#     p_img[1,:] = 0
#     p_img[end,:] = 0
#     p_img[:,1] = 0
#     p_img[:,end] = 0
#     focus = p_img
# else
#     focus = img
# end
import numpy as np
import math


# TODO fix this to properly handle filter depth and number of output channels
def convolve(img, fltr, same=False, stri=1, pad=0, repfilter=False):
    """ Works for 2d and 3d images """
    # focus = np.array{eltype(img),2}  # scope outside of if block
    if np.ndim(img) == 3:
        imgd, imgx, imgy = np.shape(img)
    elif np.ndim(img) == 2:
        imgx, imgy = np.shape(img)
        imgd = 1
    else:
        print("Wrong dimensions of image file. Quitting.")
        return

    if np.ndim(fltr) == 3:
        fd, fx, fy = np.shape(fltr)
    elif np.ndim(fltr) == 2:
        fx, fy = np.shape(fltr)
        fd = 1
    else:
        print("Wrong dimensions of filter. Quitting.")
        return

    if fd != imgd:  # as a convenience we could just replicate the 2d filter...
        print("Depths of image and filter not equal. Quitting.")
        return

    if same:
        pad = math.ceil((fx - 1) / 2)

    if pad > 0:
        img = dopad(img, pad)

    # dimensions of the result of convolution
    x_out = (imgx + 2 * pad - fx) // stri + 1
    y_out = (imgy + 2 * pad - fy) // stri + 1

    # print(imgx, imgy)

    ret = np.zeros((x_out, y_out))
    if imgd > 1:  # slice through the depth, the zeroth (first) dimension
        for i in zip(range(x_out), range(0, imgx, stri)):
            for j in zip(range(y_out), range(0, imgy, stri)):
                ret[i[0], j[0]] = np.sum(img[:, i[1]:i[1] + fx, j[1]:j[1] +
                                             fy] * fltr)
    else:
        for i in zip(range(x_out), range(0, imgx, stri)):
            for j in zip(range(y_out), range(0, imgy, stri)):
                ret[i[0], j[0]] = np.sum(img[i[1]:i[1] + fx, j[1]:j[1] +
                                             fy] * fltr)
    return ret

"""
Tnis frequires function convolve:  all this does is cycle through multiple
filters and create a multiple channel output.channel
Some of the dimension tests are duplicated.  Should clean that up for performance.
"""
def multi_conv(img, filter_stack, same=False, stri=1, pad=0):
    if np.ndim(img) == 3:
        imgd, imgx, imgy = np.shape(img)
    elif np.ndim(img) == 2:
        imgx, imgy = np.shape(img)
        imgd = 1
    else:
        print("Wrong dimensions of image file. Quitting.")
        return

    if np.ndim(filter_stack) == 4:
        nf, fd, fx, fy = np.shape(filter_stack)
        # print(nf, fd, fx, fy)
    elif np.ndim(filter_stack) == 3:
        fd, fx, fy = np.shape(filter_stack)
        if fd % imgd != 0.0:
            print("Filters are not even multiple of image depth. Quitting.")
            return
        else:
            nf = int(fd / imgd)
            fd = int(fd / nf)
            filter_stack = filter_stack.reshape(nf, fd, fx, fy)
            # print(nf, fd, fx, fy)
    else:
        print("Wrong dimensions of filter. Quitting.")
        return

    if fd != imgd:
        print("Filter depth does not match image depth. Quitting.")
        return

    # dimensions of the output of convolution
    x_out = (imgx + 2 * pad - fx) // stri + 1
    y_out = (imgy + 2 * pad - fy) // stri + 1

    ret = np.zeros((nf, x_out, y_out))

    for i in range(nf):
        ret[i] = convolve(img, filter_stack[i], same, stri, pad)

    return ret


def dopad(img, allpad=0, rowpad=0, colpad=0):
    # if you skip allpad, you must use argnames for rowpad and/or colpad
    if np.ndim(img) == 3:
        imgd, imgx, imgy = np.shape(img)
    elif np.ndim(img) == 2:
        imgx, imgy = np.shape(img)
        imgd = 1
    else:
        print("Wrong dimensions of image. Quitting.")
        return

    t = img.dtype.type  # you can't ducktype everything: set right array type

    if allpad > 0:
        rowpad = colpad = allpad

    if imgd > 1:
        hold = []
        for i in range(imgd):
            hold.append(
                np.vstack((np.zeros((rowpad, imgy + 2 * colpad), dtype=t),
                           np.hstack((np.zeros((imgx, colpad), dtype=t),
                                      img[i, :, :],
                                      np.zeros((imgx, colpad), dtype=t))),
                           np.zeros((rowpad, imgy + 2 * colpad), dtype=t))))
        return np.stack(hold, axis=0)
    else:
        return np.vstack((np.zeros((rowpad, imgy + 2 * colpad), dtype=t),
                          np.hstack((np.zeros((imgx, colpad), dtype=t),
                                     img,
                                     np.zeros((imgx, colpad), dtype=t))),
                          np.zeros((rowpad, imgy + 2 * colpad), dtype=t)))


def pool(img, szout, poolfunc=np.max):
    # poolfunc should be np.max or np.mean
    imgx, imgy = img.shape
    if (imgx % szout != 0) | (imgy % szout != 0):
        print("Pooling size does not fit evenly into image size. Quitting.")
        return
    xout = imgx // szout
    yout = imgy // szout
    ret = np.zeros((xout, yout))
    for i in range(xout):
        for j in range(yout):
            xst = i * szout
            xend = (i + 1) * szout
            yst = j * szout
            yend = (j + 1) * szout
            ret[i, j] = poolfunc(img[xst:xend, yst:yend])
    return ret


def dopad_comp(arr, pad):  # use array comprehension
    """
    a complicated one-liner:  yuck--and it's 6 times slower in Python
    """
    m, n = np.shape(arr)
    ret = np.array([0 if (i in range(pad)) or (j in range(pad)) or
                    (i in range(m + pad, m + 2 * pad)) or
                    (j in range(n + pad, n + 2 * pad))
                    else arr[i - pad, j - pad]
                    for i in range(m + 2 * pad)
                    for j in range(n + 2 * pad)])
    return ret.reshape((m + 2 * pad,
                        n + 2 * pad))


# data and filters to play with
x = np.array([[3, 0, 1, 2, 7, 4],
              [1, 5, 8, 9, 3, 1],
              [2, 7, 2, 5, 1, 3],
              [0, 1, 3, 1, 7, 8],
              [4, 2, 1, 6, 2, 8],
              [2, 4, 5, 2, 3, 9]])

x3d = np.stack((x, x, x), axis=0)

v_edge_fil = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

v_edge_fil3d = np.stack((v_edge_fil, v_edge_fil, v_edge_fil), axis=0)

h_edge_fil = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]])

sobel_fil = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

schorr_fil = np.array([[3, 0, -3],
                       [10, 0, -10],
                       [3, 0, -3]])

edg = np.array([[10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0]])

edg3d = np.stack((edg, edg, edg), axis=0)

edg2 = np.array([[10, 10, 10, 0, 0, 0],
                 [10, 10, 10, 0, 0, 0],
                 [10, 10, 10, 0, 0, 0],
                 [0, 0, 0, 10, 10, 10],
                 [0, 0, 0, 10, 10, 10],
                 [0, 0, 0, 10, 10, 10]])


# old code fragments
# from multi_conv:
    # if filter_stack.ndim == 4:
    #     for i in range(nf):
    #         ret[i] = convolve(img, filter_stack[i], same, stri, pad)
    # elif filter_stack.ndim == 3:
    #     for i in range(nf):
    #         ret[i] = convolve(img, filter_stack[i * fd:i * fd + fd],
    #                           same, stri, pad)
    # else:
    #     print("Filters are not even multiple of image depth. Quitting.")
    #     return
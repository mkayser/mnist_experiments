import numpy as np
import matplotlib.pyplot as plt


def compute_fft_images(batch):
    # compute 2D FFT over each image in one shot:
    F = np.fft.fft2(batch, axes=(1,2), norm='ortho')     # shape (N,28,28), complex
    Fshift = np.fft.fftshift(F, axes=(1,2))
    magnitude = np.log1p(np.abs(Fshift))   # shape (N,28,28) real
    # now magnitude[i] is the FFT‐magnitude image for batch[i]
    return magnitude,Fshift


def compute_fft_cell_freqs(shape):
    assert(len(shape)==2)
    center_i, center_j = shape[0]//2, shape[1]//2
    
    def distance_from_center(i,j):
        distx = i - center_i
        disty = j - center_j
        return np.sqrt(distx**2 + disty**2)

    distance_array = np.fromfunction(distance_from_center, shape, dtype=np.float32)
    return distance_array




def display_mnist_inputs_and_outputs(input_imgs, output_imgs, dims=(None,8)):
    def ceil_int_div(a,b):
        return (a+b-1) // b

    def place_image(ax, img, caption, cmap):
        ax.set_aspect('equal')
        ax.set_title(caption)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')

    def place_cumerr_plot(ax, perfreqbin_sqerr, distance_array):
        x = distance_array.flatten()
        y = perfreqbin_sqerr.flatten()

        #print("Errs={}\n".format(x))
        #print("Distances={}\n".format(y))
        
        unique_xs, inv_idx = np.unique(x, return_inverse=True)
        sums = np.bincount(inv_idx, weights=y)
        cum_error = np.cumsum(sums)

        ax.plot(unique_xs, cum_error, marker='o')


    # Make sure we have squeezed the channel dimension
    assert(len(input_imgs.shape)==3)
    assert(input_imgs.shape == output_imgs.shape)
    
    input_fft_imgs, input_complex_fft_imgs = compute_fft_images(input_imgs)
    output_fft_imgs, output_complex_fft_imgs  = compute_fft_images(output_imgs)

    perfreqbin_sqerr = np.abs(input_complex_fft_imgs - output_complex_fft_imgs) ** 2.0
    distance_array = compute_fft_cell_freqs(input_imgs.shape[1:])
    
    image_group_size = 5
    n_groups = len(input_imgs)
    n_images = n_groups * image_group_size
    
    # establish grid dimensions
    assert(len(dims)==2)
    n_rows, n_cols = dims
    assert(not(n_rows is None and n_cols is None))
    if(n_rows is None):
        n_rows = ceil_int_div(n_images, n_cols)
    if(n_cols is None):
        n_cols = ceil_int_div(n_images, n_rows)
        if(n_cols % image_group_size != 0):
            n_cols += (image_group_size - (n_cols % image_group_size))
    assert n_rows * n_cols >= n_images, "NRows={},  NCols={},   NImages={}".format(n_rows, n_cols, n_images)
    assert(n_cols % 2 == 0)

    # Build figure and axes grid
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8, 8),  # total inches
                             dpi=100,         # dots per inch
                             constrained_layout=True)

    # Populate each pair of cells
    for i in range(n_groups):
        input_img = input_imgs[i]
        output_img = output_imgs[i]
        input_fft_image = input_fft_imgs[i]
        output_fft_image = output_fft_imgs[i]
        perfreqbin_sqerr_curr = perfreqbin_sqerr[i]
        
        
        j = i * image_group_size
        r = j // n_cols
        c = j % n_cols
        assert(c+1 < n_cols)
        assert(c+2 < n_cols)
        assert(c+3 < n_cols)
        assert(c+4 < n_cols)
        place_image(axes[r][c], input_img, "Input", "gray")
        place_image(axes[r][c+1], output_img, "Output", "gray")
        place_image(axes[r][c+2], input_fft_image, "Input_FFT", "magma")
        place_image(axes[r][c+3], output_fft_image, "Output_FFT", "magma")
        place_cumerr_plot(axes[r][c+4], perfreqbin_sqerr_curr, distance_array)

    plt.show()




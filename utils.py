import numpy as np
import torch
from skimage.transform import rotate


def randomly_rotate_tile(tile, delta_rotation=45.0):
    """
    randomly rotate tile by 360/delta_rotation permutations

    :param delta_rotation: Angle in degrees, of which the rotated tile is a factor of
    :param tile: 2d contour fragment

    :return: rotated tile. Note this is an RGB format and values range b/w [0, 255]
    """
    num_possible_rotations = 360 // delta_rotation
    return rotate(tile, angle=(np.random.randint(0, np.int(num_possible_rotations)) * delta_rotation))


def tile_image(img, frag, insert_loc_arr, rotate_frags=True, delta_rotation=45, gaussian_smoothing=True,
               sigma=4.0, replace=True):
    """
    Place tile 'fragments' at the specified starting positions (x, y) in the image.

    :param replace: If True, will replace image pixels with the tile. If False will multiply
           image pixels and tile values. Default = True
    :param frag: contour fragment to be inserted
    :param insert_loc_arr: array of (x,y) position where tiles should be inserted
    :param img: image where tiles will be placed
    :param rotate_frags: If true each tile is randomly rotated before insertion.
    :param delta_rotation: min rotation value
    :param gaussian_smoothing: If True, each fragment is multiplied with a Gaussian smoothing
            mask to prevent tile edges becoming part of stimuli [they will lie in the center of the RF of
            many neurons. [Default=True]
    :param sigma: Standard deviation of gaussian smoothing mask. Only used if gaussian smoothing is True

    :return: tiled image
    """
    tile_len = frag.shape[0]

    if insert_loc_arr.ndim == 1:
        x_arr = [insert_loc_arr[0]]
        y_arr = [insert_loc_arr[1]]
    else:
        x_arr = insert_loc_arr[:, 0]
        y_arr = insert_loc_arr[:, 1]

    for idx in range(len(x_arr)):

        # print("Processing Fragment @ (%d,%d)" % (x_arr[idx], y_arr[idx]))

        if (-tile_len < x_arr[idx] < img.shape[0]) and (-tile_len < y_arr[idx] < img.shape[1]):

            start_x_loc = np.int(max(x_arr[idx], 0))
            stop_x_loc = np.int(min(x_arr[idx] + tile_len, img.shape[0]))

            start_y_loc = np.int(max(y_arr[idx], 0))
            stop_y_loc = np.int(min(y_arr[idx] + tile_len, img.shape[1]))

            # print("Placing Fragment at location  l1=(%d, %d), y = (%d, %d),"
            #       % (start_x_loc, stop_x_loc, start_y_loc, stop_y_loc))

            # Adjust incomplete beginning tiles
            if x_arr[idx] < 0:
                tile_x_start = tile_len - (stop_x_loc - start_x_loc)
            else:
                tile_x_start = 0

            if y_arr[idx] < 0:
                tile_y_start = tile_len - (stop_y_loc - start_y_loc)
            else:
                tile_y_start = 0
            #
            # print("Tile indices x = (%d,%d), y = (%d, %d)" % (
            #       tile_x_start, tile_x_start + stop_x_loc - start_x_loc,
            #       tile_y_start, tile_y_start + stop_y_loc - start_y_loc))

            if rotate_frags:
                tile = randomly_rotate_tile(frag, delta_rotation)
            else:
                tile = frag

            # multiply the file with the gaussian smoothing filter
            # The edges between the tiles will lie within the stimuli of some neurons.
            # to prevent these prom being interpreted as stimuli, gradually decrease them.
            if gaussian_smoothing:
                g_kernel = get_2d_gaussian_kernel((tile_len, tile_len), sigma=sigma)
                g_kernel = np.reshape(g_kernel, (g_kernel.shape[0], g_kernel.shape[1], 1))
                g_kernel = np.repeat(g_kernel, 3, axis=2)

                tile = tile * g_kernel

            # only add the parts of the fragments that lie within the image dimensions
            if replace:
                new_img_pixels = \
                    tile[tile_x_start: tile_x_start + stop_x_loc - start_x_loc,
                         tile_y_start: tile_y_start + stop_y_loc - start_y_loc, :]
            else:
                new_img_pixels = \
                    img[start_x_loc: stop_x_loc, start_y_loc: stop_y_loc, :] * \
                    tile[tile_x_start: tile_x_start + stop_x_loc - start_x_loc,
                         tile_y_start: tile_y_start + stop_y_loc - start_y_loc, :]

            img[start_x_loc: stop_x_loc, start_y_loc: stop_y_loc, :] = new_img_pixels

    return img


def get_2d_gaussian_kernel(shape, sigma=1.0):
    """
    Returns a 2d (unnormalized) Gaussian kernel of the specified shape.

    :param shape: x,y dimensions of the gaussian
    :param sigma: standard deviation of generated Gaussian
    :return:
    """
    ax = np.arange(-shape[0] // 2 + 1, shape[0] // 2 + 1)
    ay = np.arange(-shape[1] // 2 + 1, shape[1] // 2 + 1)
    # ax = np.linspace(-1, 1, shape[0])
    # ay = np.linspace(-1, 1, shape[1])

    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    kernel = kernel.reshape(shape)

    return kernel


class PunctureImage(object):
    """
    This is an input transform
    Add random occlusion bubbles to image.
    REF: Gosselin and Schyns - 2001 - Bubbles: a technique to reveal the use of information in
         recognition tasks

    This is actually the opposite of the reference technique, instead of masking the image and
    then revealing parts of it through bubbles. Masked out gaussian bubbles are added to the image.

    :param n_bubbles: number of bubbles
    :param fwhm: bubble full width half magnitude. Note the actual tile size is 2 * fwhm
    :param peak_bubble_transparency: 0 = Fully opaque at center (default), 1= Fully visible
        at center (no occlusion at all)

    """

    def __init__(self, n_bubbles=0, fwhm=11, tile_size=None, peak_bubble_transparency=0):

        if 0 > peak_bubble_transparency or 1 < peak_bubble_transparency:
            raise Exception("Bubble transparency {}, should be between [0, 1]".format(
                peak_bubble_transparency))
        self.peak_bubble_transparency = peak_bubble_transparency

        self.n_bubbles = n_bubbles

        self.fwhm = fwhm  # full width half magnitude
        self.bubble_sigma = fwhm / 2.35482

        if tile_size is not None:
            self.tile_size = tile_size
        else:
            if isinstance(fwhm, np.ndarray):
                max_fwhm = max(fwhm)
                self.tile_size = np.array([np.int(2 * max_fwhm), np.int(2 * max_fwhm)])
            else:
                self.tile_size = np.array([np.int(2 * self.fwhm), np.int(2 * self.fwhm)])

    def __call__(self, img, start_loc_arr=None):
        """
        Args:
            img (Tensor) : Tensor image of size (C, H, W) to be normalized.
            start_loc_arr: Starting location of the full tile. If star location
            is specified it will only add as many bubbles as the length of the
            start_location array.

        Returns:
            Tensor: Normalized Tensor image.
        """
        _, h, w = img.shape

        img = img.permute(1, 2, 0)

        if start_loc_arr is not None:
            n_bubbles = len(start_loc_arr)
        else:
            n_bubbles = self.n_bubbles
            start_loc_arr = np.array([
                np.random.randint(h - self.tile_size[0], size=self.n_bubbles),
                np.random.randint(w - self.tile_size[1], size=self.n_bubbles),
            ]).T

        if isinstance(self.bubble_sigma, np.ndarray):
            sigma_arr = np.random.choice(self.bubble_sigma, size=n_bubbles)
        else:
            sigma_arr = np.ones(n_bubbles) * self.bubble_sigma

        mask = torch.ones_like(img)
        for start_loc_idx, start_loc in enumerate(start_loc_arr):
            bubble_frag = get_2d_gaussian_kernel(
                shape=self.tile_size, sigma=sigma_arr[start_loc_idx])
            bubble_frag = torch.from_numpy(bubble_frag)
            bubble_frag = bubble_frag.float().unsqueeze(-1)
            bubble_frag = 1 - bubble_frag

            mask = tile_image(
                mask,
                bubble_frag,
                start_loc,
                rotate_frags=False,
                gaussian_smoothing=False,
                replace=False
            )

        mask = mask + self.peak_bubble_transparency
        mask[mask > 1] = 1

        ch_means = img.mean(dim=[0, 1])  # Channel Last
        # print("Channel means  {}".format(ch_means))

        masked_img = mask * img + (1 - mask) * ch_means * torch.ones_like(img)

        # # Debug
        # import matplotlib.pyplot as plt
        # plt.ion()
        #
        # f, ax_arr = plt.subplots(1, 3)
        # ax_arr[0].imshow(img)
        # ax_arr[0].set_title("Original Image")
        # ax_arr[1].imshow(masked_img)
        # ax_arr[1].set_title("Punctured Image")
        # ax_arr[2].imshow(mask)
        # ax_arr[2].set_title("Mask")
        #
        # import pdb
        # pdb.set_trace()

        return masked_img.permute(2, 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(n_bubbles={}, fwhm = {}, bubbles_sigma={}, tile_size={}), ' \
               'max bubble transparency={}'.format(
                   self.n_bubbles, self.fwhm, self.bubble_sigma, self.tile_size,
                   self.peak_bubble_transparency)
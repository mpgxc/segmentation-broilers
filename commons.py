import numpy as np
import statistics
import tifffile
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage import (filters,
                     measure,
                     exposure,
                     morphology)
from tqdm import tqdm
from glob import glob

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plot(arr_images=[], grid=(1, 1), cmap="inferno"):

    fig = plt.figure(figsize=(20, 10))

    grid = ImageGrid(fig, 111,
                   nrows_ncols=grid,
                   axes_pad=0.1)

    for ax, img in zip(grid, arr_images):
        ax.imshow(img, cmap)
        ax.axis('off')

    plt.show()

def binarize_image(arr):
    return arr > filters.threshold_triangle(arr)

def rescale_arr(arr, scale=255):
    return (arr * scale).astype('uint8')

def load_paths_images_sorted(path):

    """
    Carrega caminho da simagens e ordena em onrdem crescente os frames.
    """

    arr = []

    def parser_image_name(image_name):

        *_, name = image_name.split("/")
        name, *_ = name.split(".")

        return int(name)



    for index in tqdm(glob(f'{path}/*')):
        try:
            image_name = parser_image_name(index)

            arr.append(image_name)

        except: continue

    image_path_sorted = sorted(arr)

    image_unique_name = lambda x: f"{path}/{x}.tif"

    return list(map(image_unique_name, image_path_sorted))


def load_images_from_paths(arr_paths, is_gray=False):
  arr_images = []

  if is_gray:
    for img_path in tqdm(arr_paths):

      try:
        frame = rgb2gray(tifffile.imread(img_path))

        is_valid_frame = statistics.mode(binarize_image(frame).flatten())

        if not is_valid_frame:
            continue

        arr_images.append(rescale_arr(frame))
      except: continue
  else:
    for img_path in tqdm(arr_paths):
      try:
        frame = tifffile.imread(img_path)

        is_valid_frame = statistics.mode(binarize_image(frame).flatten())

        if not is_valid_frame:
            continue

        arr_images.append(frame)
      except: continue
  return np.asarray(arr_images)


def auto_invert_image_mask(arr):

    """
    Calcula os pixels da imagem e inverte os pixels da imagem caso os pixels True > False
    Isso é uma forma de garatir que as mascaras tenham sempre o fundo preto = 0 e o ROI = 1
    """

    img = arr.copy()

    if statistics.mode(img.flatten()):
        img = np.invert(img)

    return img

def find_bighest_cluster_area(clusters):
    regions = measure.regionprops(clusters)

    all_areas = map(lambda item: item.area, regions)

    return max(all_areas)

def find_bighest_cluster(img):

    clusters = auto_invert_image_mask(img)

    clusters = measure.label(clusters, background=0)

    cluster_size = find_bighest_cluster_area(clusters)

    return morphology.remove_small_objects(clusters,
                                         min_size=(cluster_size - 1),
                                         connectivity=8)


def check_colision_border(mask):

    x, *_ = mask.shape

    left = mask[:1, ].flatten()
    right = mask[x - 1: x, ].flatten()
    top = mask[:, : 1].flatten()
    bottom = mask[:, x - 1: x].flatten()

    borders_flatten = [left, right, top, bottom]

    if np.concatenate(borders_flatten).sum():
        return True

    return False


def rule_of_three_percent_pixels(arr):

    def co_occurrence(arr):
        unique, counts = np.unique(arr, return_counts=True)

        return dict(zip(unique, counts))

    def ternary(value):
        return 0 if value is None else value

    def binarize_image(arr):
        return arr > filters.threshold_minimum(arr)

    image_bin = binarize_image(arr)
    image_coo = co_occurrence(image_bin)

    true_value = ternary(image_coo.get(True))
    false_value = ternary(image_coo.get(False))

    _100 = false_value + true_value

    return dict({
      'true_pixels': int((true_value * 100) / _100),
      'false_pixels': int((false_value * 100) / _100)
    })

def crop_image_box(image, shape=(100, 100), margin_pixel=30):

    x, y = shape

    return image[x - margin_pixel:
               x + margin_pixel,
               y - margin_pixel:
               y + margin_pixel]

def find_center_mask(image_bin):

    props, *_ = measure.regionprops(
      measure.label(image_bin)
    )

    x, y = props.centroid

    return int(x), int(y)

def find_roi(img):

    binary_img = binarize_image(exposure.equalize_hist(img))

    best_cluster = find_bighest_cluster(binary_img)

    merged = binarize_image(binary_img + best_cluster)

    return binarize_image(find_bighest_cluster(merged))

def smoothing_mask_edges(mask):
    return binarize_image(filters.gaussian(mask, sigma=0.5))

def fill_smoothing_mask_edges(mask):

    mask = binarize_image(mask)

    mask = morphology.closing(mask, morphology.disk(9))

    mask = ndi.binary_fill_holes(mask)

    mask = filters.gaussian(mask, sigma=0.5)

    mask = binarize_image(mask)

    return find_bighest_cluster(mask)
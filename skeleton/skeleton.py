import math
import numpy as np
from PIL import Image

from .. import utils

def gravity_center(mask):
    mean_x = 0
    mean_y = 0
    count = 0
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y][x]:
                mean_x += x
                mean_y += y
                count += 1
    return mean_x / count, mean_y / count

def _coordinates_valid(x, y, width, height):
    return x > -1 and x < width and y > -1 and y < height

def _neighbours_coordinates(x, y, width, height):
    coordinates = [((i % 3 - 1 + x), (i // 3 - 1 + y)) for i in range(9) if i != 4]
    return [(x, y) for x, y in coordinates if _coordinates_valid(x, y, width, height)]

def _neighbours_count(mask):
    result = np.zeros(mask.shape)
    width, height = mask.shape[1], mask.shape[0]
    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            count = sum([mask[y][x] for x, y in _neighbours_coordinates(x, y, width, height)])
            result[y][x] = count
    return result

def _endpoints_in_mask(mask):
    endpoints = []
    mask_neighbours_count = _neighbours_count(mask)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask_neighbours_count[y][x] == 1:
                endpoints.append((x, y))
    return endpoints

def localize_apices_engine(image, mt_masks, debug, output_dir, image_name):
    from skimage.morphology.convex_hull import convex_hull_image
    from skimage.morphology import skeletonize

    if len(mt_masks) < 2:
        raise ValueError('can not infer endpoints with less than two MT masks')
    mt_masks = utils._pick_greatest(mt_masks, 2, lambda mask: np.sum(mask))

    _, _, rotated_image, rotated_mt_masks = utils._rotate_tooth_image(image, mt_masks)
    apical_mt_mask, coronal_mt_mask = rotated_mt_masks
    image = rotated_image.convert('RGB')
    width, height = image.size
    utils._apply_mask_to_image(image, apical_mt_mask, (0, 255, 0), alpha=0.05)
    utils._apply_mask_to_image(image, coronal_mt_mask, (0, 255, 0), alpha=0.05)

    apical_skeleton = skeletonize(apical_mt_mask)
    coronal_skeleton = skeletonize(coronal_mt_mask)
    utils._apply_mask_to_image(image, apical_skeleton, (255, 0, 0))
    utils._apply_mask_to_image(image, coronal_skeleton, (0, 0, 255))

    coronal_endpoints = _endpoints_in_mask(coronal_skeleton)
    apical_endpoints = _endpoints_in_mask(apical_skeleton)
    for point in coronal_endpoints + apical_endpoints:
        utils._add_mark(image, point, (255, 255, 255))

    outer_left_endpoint = utils._find_closest_point((0, height), coronal_endpoints)
    coronal_endpoints.remove(outer_left_endpoint)
    outer_right_endpoint = utils._find_closest_point((width, height), coronal_endpoints)
    inner_left_endpoint = utils._find_closest_point(outer_left_endpoint, apical_endpoints)
    inner_right_endpoint = utils._find_closest_point(outer_right_endpoint, apical_endpoints)

    utils._add_mark(image, outer_left_endpoint, (255, 128, 0)) 
    utils._add_mark(image, inner_left_endpoint, (255, 0, 0)) 
    utils._add_mark(image, outer_right_endpoint, (255, 128, 0)) 
    utils._add_mark(image, inner_right_endpoint, (255, 0, 0)) 

    return { 'output_image': image }

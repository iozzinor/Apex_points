from .. import utils
from . import raytracing_lookup

from PIL import ImageDraw
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from skimage.morphology.convex_hull import convex_hull_image
from skimage.measure import find_contours
import numpy as np

import functools
import itertools
import math
import operator

def display_outlines(rotated_image, coronal_mt_mask, apical_mt_mask):
    coronal_outline = np.array(sorted(find_contours(coronal_mt_mask), key=functools.cmp_to_key(lambda a, b: len(a) - len(b)))[-1], dtype=int)
    coronal_outline_mask = np.zeros_like(coronal_mt_mask)
    coronal_outline_mask[coronal_outline[:,0], coronal_outline[:,1]] = 1
    utils._apply_mask_to_image(rotated_image, coronal_outline_mask, (255, 0, 0), alpha=1)
    apical_outline = np.array(sorted(find_contours(apical_mt_mask), key=functools.cmp_to_key(lambda a, b: len(a) - len(b)))[-1], dtype=int)
    apical_outline_mask = np.zeros_like(apical_mt_mask)
    apical_outline_mask[apical_outline[:,0], apical_outline[:,1]] = 1
    utils._apply_mask_to_image(rotated_image, apical_outline_mask, (0, 0, 255), alpha=1)
    return [(x, y) for y, x in coronal_outline], [(x, y) for y, x in apical_outline], coronal_outline_mask, apical_outline_mask

def get_temporary_endpoints(rotated_image, thin_cancel_mask, hull):
    width, height = rotated_image.size
    temporary_endpoints = [(x, y) for y, x in utils._endpoints_in_mask(thin_cancel_mask)]
    temporary_endpoints = [point for point in temporary_endpoints if len([neighbour for neighbour in utils._neighbours_coordinates(point[0], point[1], width, height) if hull[neighbour[1]][neighbour[0]]]) < 8]
    for point in temporary_endpoints:
        utils._add_mark(rotated_image, point, (255, 255, 255))
    return temporary_endpoints

def get_dp_points(width, height, rotated_image, thin_cancel_mask, apical_outline_mask, apical_mt_mask, coronal_outline_mask, coronal_mt_mask):
    lookup = raytracing_lookup.RaytracingLookup(width, height)
    coronal_dp_points = []
    apical_dp_points = []
    for i in range(1, 360, 10):
        angle = i / 180 * math.pi
        for y, x in zip(*np.where(skeletonize(thin_cancel_mask))):
            for point in lookup.radius_line_for_angle(x, y, angle):
                if apical_outline_mask[point[1]][point[0]]:
                    if not point in apical_dp_points:
                        apical_dp_points.append(point)
                    break
                if apical_mt_mask[point[1]][point[0]]:
                    break
                if coronal_outline_mask[point[1]][point[0]]:
                    if not point in coronal_dp_points:
                        coronal_dp_points.append(point)
                    break
                if coronal_mt_mask[point[1]][point[0]]:
                    break
    for point in apical_dp_points + coronal_dp_points:
        rotated_image.putpixel(point, (255, 255, 255))
    return coronal_dp_points, apical_dp_points

def get_dentin_pulp_split_points(rotated_image, coronal_mt_mask, apical_mt_mask, coronal_outline, apical_outline, coronal_outline_mask, apical_outline_mask):
    coronal_gc = np.flip(utils._mask_gravity_center(coronal_mt_mask))
    utils._add_mark(rotated_image, coronal_gc, (255, 0, 0))
    apical_gc = np.flip(utils._mask_gravity_center(apical_mt_mask))
    utils._add_mark(rotated_image, apical_gc, (0, 0, 255))
    
    apical_split_point, coronal_split_point = None, None
    middle = ((coronal_gc[0] + apical_gc[0]) / 2, (coronal_gc[1] + apical_gc[1]) / 2)

    coronal_to_apical_1 = utils._line_points(coronal_gc, apical_gc)
    coronal_to_apical_2 = utils._line_points((coronal_gc[0]+1, coronal_gc[1]), (apical_gc[0]+1, apical_gc[1]))
    split_line_points = [[point_1, point_2] for point_1, point_2 in zip(coronal_to_apical_1, coronal_to_apical_2)]
    split_line_points = [point for sublist in split_line_points for point in sublist]
    for point in reversed(split_line_points):
        if coronal_outline_mask[point[1]][point[0]]:
            coronal_split_point = point
            break
    for point in split_line_points:
        if apical_outline_mask[point[1]][point[0]]:
            apical_split_point = point
            break
    if apical_split_point == None:
        apical_split_point = utils._find_closest_point(middle, apical_outline)
    if coronal_split_point == None:
        coronal_split_point = utils._find_closest_point(middle, coronal_outline)

    return apical_split_point, coronal_split_point

def reorder_dentin_pulp_points(dp_points, outline_points):
    dentin_pulp_indices = list(range(len(outline_points)))
    groups = [(is_dentin, list(indexes)) for is_dentin, indexes in itertools.groupby(dentin_pulp_indices, lambda i: tuple(outline_points[i]) in dp_points)]
    non_dentin_indexes = [indexes for is_dentin, indexes in groups if not is_dentin]
    if len(non_dentin_indexes) > 1 and non_dentin_indexes[-1][-1] == len(outline_points) - 1 and non_dentin_indexes[0][0] == 0:
        first = non_dentin_indexes[0]
        non_dentin_indexes[0] = non_dentin_indexes[-1]
        non_dentin_indexes[0].extend(first)
        non_dentin_indexes.pop()
    longest = sorted(non_dentin_indexes, key=functools.cmp_to_key(lambda a, b: len(a) - len(b)))[-1]
    
    point_groups = [(is_in_longest, list(indexes)) for is_in_longest, indexes in itertools.groupby(dentin_pulp_indices, lambda i: i in longest)]
    dentin_pulp_groups = [indexes for is_in_longest, indexes in point_groups if not is_in_longest]
    indices = functools.reduce(operator.iconcat, reversed(dentin_pulp_groups), [])

    ordered_points = [outline_points[i] for i in indices]
    return [point for point in ordered_points if point in dp_points]

def remove_points_crossing(points_1, points_2, mask):
    new_points_1, new_points_2 = [], []
    for point in points_1:
        closest = utils._find_closest_point(point, points_2)
        crossing_count = 0
        for potential_point in utils._line_points(point, closest):
            if mask[potential_point[1]][potential_point[0]]:
                crossing_count += 1
            if crossing_count > 2:
                break
        if crossing_count < 3:
            new_points_1.append(point)
    for point in points_2:
        closest = utils._find_closest_point(point, points_1)
        crossing_count = 0
        for potential_point in utils._line_points(point, closest):
            if mask[potential_point[1]][potential_point[0]]:
                crossing_count += 1
            if crossing_count > 2:
                break
        if crossing_count < 3:
            new_points_2.append(point)
    return new_points_1, new_points_2

def clean_dp_points(rotated_image, temporary_endpoints, apical_split_point, coronal_split_point, ordered_apical_dp_points, ordered_coronal_dp_points, mt_merged_mask):
    left_apical_points = []
    right_apical_points = []
    left_coronal_points = []
    right_coronal_points = []

    # apical
    # -------
    add_to_left = False
    for point in ordered_apical_dp_points:
        if not add_to_left and point == apical_split_point:
            add_to_left = True
            continue
        if add_to_left:
            left_apical_points.append(point)
        else:
            right_apical_points.append(point)

    left_gc = utils._gravity_center(left_apical_points)
    right_gc = utils._gravity_center(right_apical_points)
    if left_gc[0] > right_gc[0]:
        left_apical_points, right_apical_points = right_apical_points, left_apical_points

    # coronal
    # -------
    add_to_left = False
    for point in ordered_coronal_dp_points:
        if not add_to_left and point == coronal_split_point:
            add_to_left = True
            continue
        if add_to_left:
            left_coronal_points.append(point)
        else:
            right_coronal_points.append(point)
    left_gc =  utils._gravity_center(left_coronal_points)
    right_gc = utils._gravity_center(right_coronal_points)
    if left_gc[0] > right_gc[0]:
        left_coronal_points, right_coronal_points = right_coronal_points, left_coronal_points

    left_coronal_points, left_apical_points = remove_points_crossing(left_coronal_points, left_apical_points, mt_merged_mask)
    right_coronal_points, right_apical_points = remove_points_crossing(right_coronal_points, right_apical_points, mt_merged_mask)

    for point in left_coronal_points:
        rotated_image.putpixel(point, (255, 0, 255))
    for point in right_coronal_points:
        rotated_image.putpixel(point, (0, 255, 0))
    for point in left_apical_points:
        rotated_image.putpixel(point, (0, 255, 255))
    for point in right_apical_points:
        rotated_image.putpixel(point, (255, 255, 0))
    return left_coronal_points, left_apical_points, right_apical_points, right_coronal_points

def select_endpoint_pairs(left_middle_endpoint, right_middle_endpoint, left_coronal_points, left_apical_points, right_apical_points, right_coronal_points):
    if len(left_apical_points) < 1:
        raise Exception('can not find left apex due to empty left apical points array')
    if len(right_apical_points) < 1:
        raise Exception('can not find left apex due to empty right apical points array')
    if len(left_coronal_points) < 1:
        raise Exception('can not find left apex due to empty left coronal points array')
    if len(right_coronal_points) < 1:
        raise Exception('can not find left apex due to empty right coronal points array')
    left_apical = utils._find_closest_point(left_middle_endpoint, [left_apical_points[0], left_apical_points[-1]])
    left_coronal = utils._find_closest_point(left_middle_endpoint, [left_coronal_points[0], left_coronal_points[-1]])
    right_apical = utils._find_closest_point(right_middle_endpoint, [right_apical_points[0], right_apical_points[-1]])
    right_coronal = utils._find_closest_point(right_middle_endpoint, [right_coronal_points[0], right_coronal_points[-1]])
    return ((left_apical, left_coronal), (right_apical, right_coronal))

def localize_apices_engine(image, mt_masks, debug=False, output_dir='.', image_name=''):
    # load the masks

    # check the mask length
    if len(mt_masks) < 2:
        raise ValueError('2 MT masks are needed to infer')
    mt_masks = utils._find_greatest_mt_masks(mt_masks)
    angle, _, rotated_image, mt_masks = utils._rotate_tooth_image(image, mt_masks)
    width, height = rotated_image.size

    apical_mt_mask, coronal_mt_mask = mt_masks
    mt_merged_mask = apical_mt_mask | coronal_mt_mask

    # gradient cancel masks
    coronal_distance = ndi.distance_transform_edt(np.where(coronal_mt_mask > 0, 0, 1))
    apical_distance = ndi.distance_transform_edt(np.where(apical_mt_mask > 0, 0, 1))
    hull = convex_hull_image(apical_mt_mask | coronal_mt_mask)
    cancel_mask = np.where(np.abs(coronal_distance - apical_distance) < 2, 1, 0)
    thin_cancel_mask = skeletonize(cancel_mask) & hull
    apical_hull = convex_hull_image(apical_mt_mask)

    utils._apply_mask_to_image(rotated_image, cancel_mask, (0, 255, 0))
    utils._apply_mask_to_image(rotated_image, thin_cancel_mask, (255, 255, 0), alpha=1)
    utils._apply_mask_to_image(rotated_image, coronal_mt_mask, (255, 0, 0), alpha=0.2)
    utils._apply_mask_to_image(rotated_image, apical_hull, (0, 0, 255), alpha=0.2)

    # display outlines
    coronal_outline, apical_outline, coronal_outline_mask, apical_outline_mask = display_outlines(rotated_image, coronal_mt_mask, apical_mt_mask)

    # display hull
    utils._apply_mask_to_image(rotated_image, hull, (255, 0, 255), alpha=0.05)

    # temporary endpoints
    temporary_endpoints = get_temporary_endpoints(rotated_image, thin_cancel_mask, hull)
    if len(temporary_endpoints) < 2:
        _logger.error('can not find endpoints if less than 2 potential endpoints')
        return
    coronal_dp_points, apical_dp_points = get_dp_points(width, height, rotated_image, thin_cancel_mask, apical_outline_mask, apical_mt_mask, coronal_outline_mask, coronal_mt_mask)

    # get the split points
    apical_split_point, coronal_split_point = get_dentin_pulp_split_points(rotated_image, coronal_mt_mask, apical_mt_mask, coronal_outline, apical_outline, coronal_outline_mask, apical_outline_mask)
    utils._add_mark(rotated_image, apical_split_point, (255, 128, 0))
    utils._add_mark(rotated_image, coronal_split_point, (255, 128, 0))
    apical_dp_points.append(apical_split_point)
    coronal_dp_points.append(coronal_split_point)

    # reorder the points
    ordered_apical_dp_points = reorder_dentin_pulp_points(apical_dp_points, apical_outline)
    ordered_coronal_dp_points = reorder_dentin_pulp_points(coronal_dp_points, coronal_outline)
    for point in ordered_apical_dp_points + ordered_coronal_dp_points:
        rotated_image.putpixel(point, (255, 128, 0))

    # clean the dp points
    left_middle_endpoint, right_middle_endpoint = temporary_endpoints[0], temporary_endpoints[1]
    if left_middle_endpoint[0] > right_middle_endpoint[0]:
        left_middle_endpoint, right_middle_endpoint = right_middle_endpoint, left_middle_endpoint
    left_coronal_points, left_apical_points, right_apical_points, right_coronal_points = clean_dp_points(rotated_image, temporary_endpoints, apical_split_point, coronal_split_point, ordered_apical_dp_points, ordered_coronal_dp_points, mt_merged_mask)

    # find the new endpoints
    draw = ImageDraw.Draw(rotated_image)
    endpoint_pairs = select_endpoint_pairs(left_middle_endpoint, right_middle_endpoint, left_coronal_points, left_apical_points, right_apical_points, right_coronal_points)
    for endpoint_pair in endpoint_pairs:
        for point in endpoint_pair:
            utils._add_mark(rotated_image, point, (255, 0, 0))
        draw.line(endpoint_pair, (255, 0, 0))

    (tooth_height, min_y, max_y) = utils._get_tooth_height(apical_mt_mask | coronal_mt_mask)
    draw.line((width // 2, min_y, width // 2, max_y), (0, 0, 255))

    (i3m, min_apex_opening, max_apex_opening) = utils._compute_i3m(endpoint_pairs, tooth_height)

    result = { 'output_image': rotated_image, 'I3M': i3m, 'min_apex_opening': min_apex_opening, 'max_apex_opening': max_apex_opening, 'height': height }
    
    return result

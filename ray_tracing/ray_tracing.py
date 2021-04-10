from .image_merger import ImageMerger as _ImageMerger
from .. import utils

from PIL import Image, ImageDraw
import numpy as np
from skimage.measure import find_contours

import operator
import math
import json
import os
import sys

def _perform_rotation(localize_apices_shared, debug_image_merger):
    from skimage.morphology.convex_hull import convex_hull_image

    initial_height, initial_width = localize_apices_shared['initial_mt_masks'][0].shape
    mt_masks = localize_apices_shared['initial_mt_masks']
    mt_merged_mask = np.logical_or(mt_masks[0], mt_masks[1])
    localize_apices_shared['initial_mt_merged_mask'] = mt_merged_mask
    hull_mask = convex_hull_image(mt_merged_mask)

    hull_gc = utils._mask_gravity_center(hull_mask)
    hull_gc = (int(hull_gc[1]), int(hull_gc[0]))
    mt_merged_gc = utils._mask_gravity_center(mt_merged_mask)
    mt_merged_gc = (int(mt_merged_gc[1]), int(mt_merged_gc[0]))

    angle = math.atan2(mt_merged_gc[1] - hull_gc[1], mt_merged_gc[0] - hull_gc[0]) + math.pi / 2
    angle_deg = angle * 180 / math.pi

    localize_apices_shared['angle'] = angle
    localize_apices_shared['angle_deg'] = angle_deg
    localize_apices_shared['mt_masks'] = [np.where(np.array(utils._np_binary_to_pillow_image(mask).rotate(angle_deg, expand=True)), 1, 0) for mask in localize_apices_shared['initial_mt_masks']]
    localize_apices_shared['mt_merged_mask'] = np.where(np.array(utils._np_binary_to_pillow_image(mt_merged_mask).rotate(angle_deg, expand=True)), 1, 0)
    height, width = localize_apices_shared['mt_merged_mask'].shape
    localize_apices_shared['size'] = (width, height)

    hull_gc, mt_merged_gc = utils._translate_points(utils._rotate_points([hull_gc, mt_merged_gc], initial_width, initial_height, angle), ((width - initial_width) / 2, (height - initial_height) / 2))
    localize_apices_shared['mt_merged_gc'] = mt_merged_gc
    rotated_debug_image = utils._np_binary_to_pillow_image(localize_apices_shared['mt_merged_mask']).convert('RGB')
    utils._add_mark(rotated_debug_image, hull_gc)
    utils._add_mark(rotated_debug_image, mt_merged_gc, color=(0, 0, 255))

    rotated_hull_mask = np.where(np.array(utils._np_binary_to_pillow_image(hull_mask).rotate(angle_deg, expand=True)), 1, 0)
    for contours in find_contours(rotated_hull_mask):
        coordinates = contours.astype(int)
        mask = np.zeros(rotated_hull_mask.shape)
        mask[coordinates[:,0], coordinates[:,1]] = 1
        utils._apply_mask_to_image(rotated_debug_image, mask, (255, 0, 0))

    debug_image_merger.add_image(rotated_debug_image, 'Rotation with gravity centers (hull / MT)')

def _find_pulp_lines(localize_apices_shared, debug_image_merger):
    # get the outlines
    outlines = []
    for mask in localize_apices_shared['mt_masks']:
        contours = find_contours(mask)
        contour = sorted([(len(contour), contour) for contour in contours], key=operator.itemgetter(0))[-1][1]
        outlines.append(contour.astype(int))
    # sort the outlines by number of pixels
    outlines = [sorted_outline[1] for sorted_outline in sorted([(outline.shape[0], outline) for outline in outlines], key=operator.itemgetter(0))[-2:]]
    coronal_outline = outlines[1]
    apical_outline = outlines[0]
    localize_apices_shared['coronal_outline'] = coronal_outline
    localize_apices_shared['apical_outline'] = apical_outline

    width, height = localize_apices_shared['size']
    outlines_image = Image.new('RGB', (width, height))

    # draw the outlines
    mask = np.zeros((height, width))
    for outline in outlines:
        mask[:] = 0
        mask[outline[:,0], outline[:,1]] = 1
        utils._apply_mask_to_image(outlines_image, mask, utils._random_rgb_color())

    # filter points
    coronal_outline = [(x, y) for y, x in coronal_outline]
    apical_outline = [(x, y) for y, x in apical_outline]
    coronal_pulp_line, apical_pulp_line = utils._filter_shortest_distance_intersection(coronal_outline, apical_outline, localize_apices_shared['mt_merged_mask'])
    for x, y in coronal_pulp_line:
        outlines_image.putpixel((x, y), (255, 0, 0))
    for x, y in apical_pulp_line:
        outlines_image.putpixel((x, y), (255, 128, 0))

    localize_apices_shared['coronal_pulp_line'] = coronal_pulp_line
    localize_apices_shared['apical_pulp_line'] = apical_pulp_line

    # find full lines
    full_lines_image = Image.new('RGB', (width, height))
    coronal_full_pulp_line = utils._find_full_pulp_line(coronal_outline, coronal_pulp_line) 
    apical_full_pulp_line = utils._find_full_pulp_line(apical_outline, apical_pulp_line) 
    localize_apices_shared['coronal_full_pulp_line'] = coronal_full_pulp_line
    localize_apices_shared['apical_full_pulp_line'] = apical_full_pulp_line
    for x, y in coronal_full_pulp_line:
        full_lines_image.putpixel((x, y), (255, 0, 0))
    for x, y in apical_full_pulp_line:
        full_lines_image.putpixel((x, y), (255, 128, 0))
    # add marks for temporary endpoints
    for endpoint in [coronal_full_pulp_line[0], coronal_full_pulp_line[-1], apical_full_pulp_line[0], apical_full_pulp_line[-1]]:
        utils._add_mark(full_lines_image, endpoint, (255, 255, 255))

    # draw segments perpendicular to the canal axis
    coronal_length = len(coronal_full_pulp_line)
    apical_length = len(apical_full_pulp_line)
    reverse = utils._distance(coronal_full_pulp_line[0], apical_full_pulp_line[0]) > utils._distance(coronal_full_pulp_line[0], apical_full_pulp_line[-1])
    progress_ratio = 10 / apical_length
    progress = 0
    draw = ImageDraw.Draw(outlines_image)
    while progress < 1:
        coronal_index = int(progress * coronal_length)
        apical_index = int(progress * apical_length)
        if reverse:
            apical_index = apical_length - apical_index - 1
        progress += progress_ratio
        draw.line((coronal_full_pulp_line[coronal_index], apical_full_pulp_line[apical_index]), fill=(255, 255, 0))
    # draw the last one
    draw.line((coronal_full_pulp_line[-1], apical_full_pulp_line[0] if reverse else apical_full_pulp_line[-1]), fill=(255, 255, 0))

    debug_image_merger.add_image(outlines_image, 'Outlines')
    debug_image_merger.add_image(full_lines_image, 'Pulp lines')

def _split_pulp_lines(localize_apices_shared, debug_image_merger):
    width, height = localize_apices_shared['size']
    split_image = Image.new('RGB', (width, height))

    coronal_full_pulp_line = localize_apices_shared['coronal_full_pulp_line'] 
    apical_full_pulp_line = localize_apices_shared['apical_full_pulp_line'] 
    mt_merged_gc = localize_apices_shared['mt_merged_gc'] 

    outer_left_line, outer_right_line, left_split_point = utils._split_pulp_line(mt_merged_gc, coronal_full_pulp_line)
    inner_left_line, inner_right_line, right_split_point = utils._split_pulp_line(mt_merged_gc, apical_full_pulp_line)
    for x, y in outer_left_line:
        split_image.putpixel((x, y), (255, 0, 0))
    for x, y in inner_left_line:
        split_image.putpixel((x, y), (0, 255, 0))
    for x, y in outer_right_line:
        split_image.putpixel((x, y), (0, 0, 255))
    for x, y in inner_right_line:
        split_image.putpixel((x, y), (255, 255, 0))

    utils._add_mark(split_image, left_split_point, (105, 58, 0))
    utils._add_mark(split_image, right_split_point, (105, 58, 0))

    localize_apices_shared['outer_left_pulp_line'] = outer_left_line
    localize_apices_shared['inner_left_pulp_line'] = inner_left_line
    localize_apices_shared['outer_right_pulp_line'] = outer_right_line
    localize_apices_shared['inner_right_pulp_line'] = inner_right_line

    debug_image_merger.add_image(split_image, 'Split pulp lines')

def _find_endpoints(localize_apices_shared, debug_image_merger):
    width, height = localize_apices_shared['size']
    outer_left_line = localize_apices_shared['outer_left_pulp_line']
    inner_left_line = localize_apices_shared['inner_left_pulp_line']
    outer_right_line = localize_apices_shared['outer_right_pulp_line']
    inner_right_line = localize_apices_shared['inner_right_pulp_line']

    filtered_image = Image.new('RGB', (width, height))

    filtered_outer_left_line, filtered_inner_left_line = utils._filter_shortest_distance_intersection(outer_left_line, inner_left_line, localize_apices_shared['mt_merged_mask'])
    filtered_outer_right_line, filtered_inner_right_line = utils._filter_shortest_distance_intersection(outer_right_line, inner_right_line, localize_apices_shared['mt_merged_mask'])
    
    for x, y in filtered_outer_left_line:
        filtered_image.putpixel((x, y), (255, 0, 0))
    for x, y in filtered_inner_left_line:
        filtered_image.putpixel((x, y), (0, 255, 0))
    for x, y in filtered_outer_right_line:
        filtered_image.putpixel((x, y), (0, 0, 255))
    for x, y in filtered_inner_right_line:
        filtered_image.putpixel((x, y), (255, 255, 0))

    bottom_left = (0, height)
    bottom_right = (width, height)
    left_outer_endpoint = filtered_outer_left_line[0] if utils._distance(filtered_outer_left_line[0], bottom_left) < utils._distance(filtered_outer_left_line[-1], bottom_left) else filtered_outer_left_line[-1]
    left_inner_endpoint = filtered_inner_left_line[0] if utils._distance(filtered_inner_left_line[0], left_outer_endpoint) < utils._distance(filtered_inner_left_line[-1], left_outer_endpoint) else filtered_inner_left_line[-1]
    right_outer_endpoint = filtered_outer_right_line[0] if utils._distance(filtered_outer_right_line[0], bottom_right) < utils._distance(filtered_outer_right_line[-1], bottom_right) else filtered_outer_right_line[-1]
    right_inner_endpoint = filtered_inner_right_line[0] if utils._distance(filtered_inner_right_line[0], right_outer_endpoint) < utils._distance(filtered_inner_right_line[-1], right_outer_endpoint) else filtered_inner_right_line[-1]

    endpoint_pairs = [ (left_outer_endpoint, left_inner_endpoint), (right_inner_endpoint, right_outer_endpoint) ]
    for pair in endpoint_pairs:
        for endpoint in pair:
            utils._add_mark(filtered_image, endpoint, (255, 255, 255))
    localize_apices_shared['endpoint_pairs'] = endpoint_pairs

    debug_image_merger.add_image(filtered_image, 'Filtered pulp lines')

def _compute_i3m(localize_apices_shared, debug_image_merger):
    endpoint_pairs = localize_apices_shared['endpoint_pairs']
    image = localize_apices_shared['image'].convert('RGB').rotate(localize_apices_shared['angle_deg'], expand=True)
    mt_merged_gc = localize_apices_shared['mt_merged_gc']

    utils._apply_mask_to_image(image, localize_apices_shared['mt_merged_mask'], color=(0, 255, 0), alpha=0.05)

    draw = ImageDraw.Draw(image)
    for pair in endpoint_pairs:
        for point in pair:
            utils._add_mark(image, point)
        draw.line(pair, fill=(255, 0, 0))
        label_position = (min(pair[0][0], pair[1][0]), max(pair[0][1], pair[1][1]) + 30)
        draw.text(label_position, f'{distance:0.2f} pixels', fill=(255, 0, 0))

    height, min_y, max_y = utils._get_tooth_height(localize_apices_shared['mt_merged_mask'])
    draw.line((mt_merged_gc[0], min_y, mt_merged_gc[0], max_y), fill=(0, 0, 255))
    draw.text((mt_merged_gc[0] + 30, int((min_y + max_y) / 2)), f'height: {height} pixels', fill=(0, 0, 255))

    i3m = utils._compute_i3m(endpoint_pairs, height)
    draw.text((30, 30), f'I3M: {i3m:.03f}', fill=(255, 0, 0))

    localize_apices_shared['output_image'] = image
    localize_apices_shared['I3M'] = i3m

    debug_image_merger.add_image(image, 'I3M')

def localize_apices_engine(image, mt_masks, debug=False, output_dir='.', image_name=''):
    debug_image_merger = _ImageMerger()

    # load the masks
    width, height = image.size

    # check the mask length
    if len(mt_masks) < 2:
        raise ValueError('2 MT masks are needed to infer')
    mt_masks = utils._find_greatest_mt_masks(mt_masks)

    # a shared object
    localize_apices_shared = { 'initial_mt_masks': mt_masks, 'image': image }
    
    try:
        # step 1: perform rotation
        _perform_rotation(localize_apices_shared, debug_image_merger)
        # step 2: find pulp lines
        _find_pulp_lines(localize_apices_shared, debug_image_merger)
        # step 3: split the pulp lines in right and left
        _split_pulp_lines(localize_apices_shared, debug_image_merger)
        # step 4: find new endpoints by searching lines
        _find_endpoints(localize_apices_shared, debug_image_merger)
        # step 5: compute i3m
        _compute_i3m(localize_apices_shared, debug_image_merger)
    except:
        raise
    finally:
        if debug:
            output_image_name = os.path.join(output_dir, f'{image_name}-debug.png')
            image_to_save = debug_image_merger.generate_image()
            image_to_save.save(output_image_name)
    
    return localize_apices_shared

from PIL import Image, ImageDraw
import numpy as np
import operator
import math
import io
import base64

def _labelme_shape_to_mask(shape, width, height):
    """
    Parameters
    ----------
    shape: dictionary
        The shape as stored in the labelme json file
    width: int
        The image width
    height: int
        The image height

    Returns
    -------
    np.array: a binary image representing the mask
    """
    mask = Image.new('L', (width, height))
    points = [tuple(point) for point in shape['points']]
    ImageDraw.Draw(mask).polygon(points, fill=1)
    return np.array(mask)

def _labelme_annotations_to_image(annotations):
    image_data = base64.b64decode(annotations['imageData'])
    buffer_file = io.BytesIO()
    buffer_file.write(image_data)
    return Image.open(buffer_file)


def _np_binary_to_pillow_image(np_array):
    """
    Convert a numpy array (considered as a binary) to a pillow image.
    """
    return Image.fromarray(np.uint8(np.where(np_array > 0, 255, 0)))

def _find_greatest_mt_masks(masks):
    """
    If the masks list is greater than 2, filter the good ones: the ones with the highest area.
    It is is 2, sort the masks by highest area.
    """
    sorted_masks = sorted([(np.sum(mask), mask) for mask in masks], key=operator.itemgetter(0))
    return [sorted_mask[1] for sorted_mask in sorted_masks[-2:]]

def _apply_mask_to_image(image, mask, color, alpha=0.5):
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if mask[y][x]:
                pixel = image.getpixel((x, y))
                new_color = tuple([int(current_component * (1 - alpha) + new_component * alpha) for current_component, new_component in zip(pixel, color)])
                image.putpixel((x, y), new_color)

def _add_mark(image, point, color=(255, 0, 0), size=5):
    """
    Add a circle mark in the image.

    Parameters
    ----------
    image: PIL.Image
    point: (x, y)
    """
    x, y = point
    ImageDraw.Draw(image).ellipse((x - size // 2, y - size // 2, x + size // 2, y + size // 2), fill=color)

def _gravity_center(points):
    count, x_sum, y_sum = 0, 0, 0
    for x, y in points:
        count += 1
        x_sum += x
        y_sum += y
    if count == 0:
        raise ValueError('Empty array')
    return x_sum / count, y_sum / count

def _mask_gravity_center(mask):
    indices = np.where(mask > 0)
    return _gravity_center(zip(*indices))

def _translate_points(points, translation):
    """
    Translate a list of 2d points using a translation vector.

    Parameters
    ----------
    points: [(x, y)]
    translation: (x, y]

    Returns
    -------
    [(x, y)] The list of new points.
    """
    x_list, y_list = zip(*points)
    x_list = [x + translation[0] for x in x_list]
    y_list = [y + translation[1] for y in y_list]
    return zip(x_list, y_list)
        
def _rotate_points(points, width, height, angle):
    points = _translate_points(points, (-(width / 2), -(height / 2)))
    new_points = []
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    for x, y in points:
        new_x = cos_angle * x + sin_angle * y
        new_y = -sin_angle * x + cos_angle * y
        new_points.append((new_x, new_y))
    return _translate_points(new_points, (width / 2, height / 2))

def _random_rgb_color():
    """
    Returns
    -------
    (int, int, int): A randomly generated color with components in the range 0-255.
    """
    return tuple([int(c * 255) for c in np.random.random(3)])

def _distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _find_closest_point(target, points):
    """
    Search in points for the point with the lowest distance to target
    """
    result = None
    min_distance = None
    for point in points:
        distance = _distance(point, target)
        if min_distance is None or distance < min_distance:
            min_distance = distance
            result = point
    return result

def _filter_shortest_distance_intersection(points_1, points_2, mask):
    """
    For each point in the first set, find the closest point in the second one.
    If the line between these two points contains any point in the mask,
    then the point is not included in the resulting set.
    """
    filtered_points_1 = []
    filtered_points_2 = []
    for p in points_1:
        closest = _find_closest_point(p, points_2)
        line_points = _line_points(p, closest)
        points_in_mask = len([point for point in line_points if mask[point[1]][point[0]]])
        if points_in_mask > 2:
            continue
        filtered_points_1.append(p)
    for p in points_2:
        closest = _find_closest_point(p, points_1)
        line_points = _line_points(p, closest)
        points_in_mask = len([point for point in line_points if mask[point[1]][point[0]]])
        if points_in_mask > 2:
            continue
        filtered_points_2.append(p)
    return filtered_points_1, filtered_points_2

def _line_points(a, b):
    """
    Returns
    -------
    A list of points drawing the line from a to b
    """
    x1, y1 = [int(x) for x in a]
    x2, y2 = [int(y) for y in b]
    if x1 == x2:
        return [(x1, y1 + (y if y2 > y1 else -y)) for y in range(0, abs(y1 - y2) + 1)]
    elif x1 > x2:
        return _line_points(b, a)

    dx = x2 - x1
    dy = y2 - y1
    if abs(dy) > abs(dx):
        return [(p[1], p[0]) for p in _line_points((y1, x1), (y2, x2))]

    points = []
    derr = abs(dy / dx)
    err = 0.0
    y = y1
    for x in range(x1, x2+1):
        points.append((x, y))
        err += derr
        if err > 0.5:
            y += 1 if dy > 0 else -1
            err -= 1.0
    return points

def _are_neighbours(a, b):
    return abs(a[0] - b[0]) < 2 and abs(a[1] - b[1]) < 2

def _are_lines_neighbours(a, b):
    for point in a:
        for other in b:
            if _are_neighbours(point, other):
                return True
    return False

def _find_full_pulp_line(outline, pulp_line):
    segments = [[]]
    for point in outline:
        if point in pulp_line:
            if len(segments[-1]) > 0:
                segments.append([])
            continue
        else:
            segments[-1].append(point)
    if len(segments[-1]) == 0:
        segments = segments[:-1]
    
    # first and last segments are neighbours
    if len(segments) > 1 and _are_lines_neighbours(segments[0], segments[-1]):
        segments[0].extend(segments[-1])
        segments = segments[:-1]

    # find the longest
    longest = sorted([(len(segment), segment) for segment in segments], key=operator.itemgetter(0))[-1][1]

    first = []
    last = []
    append_to_first = True
    for point in outline:
        if point in longest:
            append_to_first = False
            continue
        if append_to_first:
            first.append(point)
        else:
            last.append(point)
    if len(last) > 0:
        full_line = last
        full_line.extend(first)
        return full_line
    return first
    
def _split_pulp_line(pivot_point, full_pulp_line):
    """
    Split a line by searching for the closest point of the line to the pivot point.

    Returns
    -------
    (leftmost segment, rightmost segment, split point)
    """
    split_point = _find_closest_point(pivot_point, full_pulp_line)
    first, last = [], []
    append_to_first = True
    for point in full_pulp_line:
        if point == split_point:
            append_to_first = False
            continue
        if append_to_first:
            first.append(point)
        else:
            last.append(point)
    gc_first = _gravity_center(first) 
    gc_last = _gravity_center(last) 
    if gc_first[0] > gc_last[0]:
        first, last = last, first
    return first, last, split_point

def _get_tooth_height(mt_merged_mask):
    width, height = mt_merged_mask.shape[1], mt_merged_mask.shape[0]
    min_y = None
    max_y = None
    y = 0
    while y < height and min_y is None:
        for x in range(width):
            if mt_merged_mask[y][x]:
                min_y = y
                break
        y += 1
    y = height - 1
    while y > -1 and max_y is None:
        for x in range(width):
            if mt_merged_mask[y][x]:
                max_y = y
                break
        y -= 1
    return max_y - min_y, min_y, max_y

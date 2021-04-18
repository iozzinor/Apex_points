import math

from .. import utils

class RaytracingLookup:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.size = max(width, height)
        self.radius = math.ceil(math.sqrt(2*(self.size**2)))
        self.lookup = []
        for dy in range(self.size+1):
            angle = math.asin(dy / self.radius)
            dx = int(math.cos(angle) * self.radius)
            self.lookup.append(utils._line_points((0, 0), (dx, dy)))

    def radius_line_for_angle(self, x, y, angle):
        dy = int(self.radius * math.sin(angle))
        dx = int(self.radius * math.cos(angle))
        return self.radius_line_atan2(x, y, dx, dy)

    def radius_lines_for_angles(self, x, y, start_angle, arc):
        lines = []
        for dx, dy in self._border_coordinates(start_angle, arc):
            lines.append(list(self.radius_line_atan2(x, y, dx, dy)))
        return lines

    def _border_coordinates(self, start_angle, arc):
        coordinates = [(self.width, i) for i in range(self.height+1)]
        coordinates.extend([(i, self.height) for i in range(self.width,-self.width-1,-1)])
        coordinates.extend([(-self.width, i) for i in range(self.height,-self.height-1,-1)])
        coordinates.extend([(i, -self.height) for i in range(-self.width,self.width+1)])
        coordinates.extend([(self.width, i) for i in range(-self.height,0)])

        # find the start index
        start_angle = math.atan2(math.sin(start_angle), math.cos(start_angle)) + math.pi # [ 0 ; 2π ]
        total_count = len(coordinates)
        start_index = int(total_count * start_angle / (2 * math.pi))
        coordinates_to_display = int(total_count * (arc % (2 * math.pi)) / (2 * math.pi))
        end_index = min(total_count-1, start_index+coordinates_to_display)
        for x,y in coordinates[start_index:end_index]:
            yield x, y
        if total_count - start_index < coordinates_to_display:
            for x,y in coordinates[:(coordinates_to_display - total_count + start_index)]:
                yield x, y
 
    def radius_line_atan2(self, x, y, dx, dy):
        line_points = list(self._radius_line_offset_atan2(dx, dy))
        line_points_count = len(line_points)
        index = 0
        current_x, current_y = x, y
        while utils._are_coordinates_valid(current_x, current_y, self.width, self.height) and index < line_points_count:
            yield int(current_x), int(current_y)
            current_x, current_y = x + line_points[index][0], y + line_points[index][1]
            index += 1

    def _radius_line_offset_atan2(self, dx, dy):
        if abs(dx) < abs(dy):
            for point_x, point_y in self._radius_line_offset_atan2(dy, dx):
                yield point_y, point_x
        elif dx < 0:
            for point_x, point_y in self._radius_line_offset_atan2(-dx, dy):
                yield -point_x, point_y
        elif dy < 0:
            for point_x, point_y in self._radius_line_offset_atan2(dx, -dy):
                yield point_x, -point_y
        else:
            for point in self.lookup[dy]:
                yield point


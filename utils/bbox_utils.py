from typing import List, Tuple, Dict, Any

def parse_quad_boxes(quad_boxes: List[List[float]]) -> List[Tuple[int, int, int, int]]:
    """
    Parse Florence-2 quad_boxes format to standard bounding boxes.
    Florence-2 format: [[x1, y1, x2, y1, x2, y2, x1, y2], ...]
    Output format: [(x_min, y_min, x_max, y_max), ...]
    """
    bboxes = []
    for qb in quad_boxes:
        if len(qb) != 8:
            continue
        xs = [qb[0], qb[2], qb[4], qb[6]]
        ys = [qb[1], qb[3], qb[5], qb[7]]
        x_min = int(min(xs))
        y_min = int(min(ys))
        x_max = int(max(xs))
        y_max = int(max(ys))
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

def apply_padding(bboxes: List[Tuple[int, int, int, int]], padding: int, image_width: int, image_height: int) -> List[Tuple[int, int, int, int]]:
    """
    Apply padding to bounding boxes, ensuring they stay within image boundaries.
    """
    padded_bboxes = []
    for x_min, y_min, x_max, y_max in bboxes:
        new_x_min = max(0, x_min - padding)
        new_y_min = max(0, y_min - padding)
        new_x_max = min(image_width, x_max + padding)
        new_y_max = min(image_height, y_max + padding)
        padded_bboxes.append((new_x_min, new_y_min, new_x_max, new_y_max))
    return padded_bboxes

def filter_bboxes(bboxes: List[Tuple[int, int, int, int]], min_width: int, min_height: int) -> List[Tuple[int, int, int, int]]:
    """
    Filter bounding boxes by minimum width and height.
    """
    filtered = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        if width >= min_width and height >= min_height:
            filtered.append(bbox)
    return filtered

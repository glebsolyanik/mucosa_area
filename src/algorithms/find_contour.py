import numpy as np

from core.figure import get_center_of_figure
from algorithms.average_width import get_bounds

def find_contour_points(figure):
    """Алгоритм поиска контура"""

    center = get_center_of_figure(figure)

    partition = get_partition_contour(figure, center)
    lateral_wall = get_lateral_wall_contour(figure, center)
    left_bound, right_bound = get_bounds(figure, partition, lateral_wall)

    bottom = get_bottom_contour(figure, center, left_bound, right_bound)

    return partition, lateral_wall, bottom

def get_partition_contour(fig, center):
    """Поиск контура левой фигуры"""
    mask = (fig[1] < center[1]) & (fig[0] > center[0])

    partition = fig[:, mask]

    if partition.shape[1] == 0:
        return []

    return find_contour_points(fig=partition, axis=0)

def get_lateral_wall_contour(fig, center):
    """Поиск контура правой фигуры"""
    mask = (fig[1] > center[1]) & (fig[0] > center[0])

    lateral_wall = fig[:, mask]

    if lateral_wall.shape[1] == 0:
        return []

    return find_contour_points(fig=lateral_wall, axis=0, calc_min=True)
    
def get_bottom_contour(fig, center, left_bound, right_bound):
    mask = (fig[1] >= left_bound[1]) & (fig[1] <= right_bound[1]) & (fig[0] <= center[0])
    bottom = fig[:, mask]

    if bottom.shape[1] == 0:
        return []

    return find_contour_points(fig=bottom, axis=1) 

def find_contour_points(fig, axis, calc_min=False):
    """Поиск максимальных значений противоположной оси для каждого уникального значения указанной оси"""
    
    contour = []
    for ax in np.unique(fig[axis]):
        if calc_min:
            opp_max = np.min(fig[axis ^ 1, fig[axis] == ax])
        else:
            opp_max = np.max(fig[axis ^ 1, fig[axis] == ax])
        if axis == 0:
            contour.append([ax, opp_max])
        else:
            contour.append([opp_max, ax])
    return np.array(contour).T


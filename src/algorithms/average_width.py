import numpy as np

from core.figure import get_center_of_figure

def find_middle_points(figure) -> tuple[np.ndarray]:
    """Алгоритм поиска средних точек в ширину и длину.

    1. Находим центр фигуры;
    2. Находим точки каждого сегмента.
    
    Args:
        figure - бинарное изображение фигуры двумерного среза
    
    Returns:


    """
    center = get_center_of_figure(figure)
    
    partition = get_partition_points(figure, center)
    lateral_wall = get_lateral_wall_points(figure, center)
    left_bound, right_bound = get_bounds(figure, partition, lateral_wall)

    mask = (lateral_wall[1, :] > right_bound[1])
    lateral_wall = lateral_wall[:, mask]

    bottom = get_bottom_points(figure, center, left_bound, right_bound)
    
    return (partition, lateral_wall, bottom)

def get_partition_points(figure, center) -> np.ndarray:
    """Получение середины точек по ширине для перегородки.
    
    1. Находим точки, которые левее и выше центра.
    2. Определяем серединные точки по ширине (т.к. перегородка - вертикальный сегмент)

    Args:
        figure - бинарное изображение среза
    """

    mask = (figure[1] < center[1]) & (figure[0] > center[0])

    partition = figure[:, mask]
    
    if partition.shape[1] == 0:
        return []

    return calculate_middle_points(figure=partition, axis=0)

def calculate_middle_points(figure, axis) -> np.ndarray:
    """
    Функция проходит по точкам фигуры и вычисляет среднее по заданной оси.

    axis - ось по которой вычисляется среднее значение. 1 - x, 0 - z
    """
    if axis not in [0, 1]:
        raise ValueError('Неверно выбрана ось!')

    middle_points: list[tuple[float]] = []

    for ax in np.unique(figure[axis]):
        median = np.median(figure[axis ^ 1, figure[axis] == ax])
        if axis == 0:
            middle_points.append((ax, median))
        else:
            middle_points.append((median, ax))
            
    return np.array(middle_points).T

def get_lateral_wall_points(figure, center) -> np.ndarray:
    """Получение середины точек по ширине для латеральной стенки"""
    
    mask = (figure[1] > center[1]) & (figure[0] > center[0])

    lateral_wall = figure[:, mask]
    
    if lateral_wall.shape[1] == 0:
        return []
    
    return calculate_middle_points(figure=lateral_wall, axis=0)
    
def get_bottom_points(fig, center, left_bound, right_bound) -> np.ndarray:
    """Получение середины точек по высоте для дна носа"""

    mask = (fig[1] > left_bound[1]) & (fig[1] < right_bound[1]) & (fig[0] <= center[0])
    bottom = fig[:, mask]

    if bottom.shape[1] == 0:
        return []

    return calculate_middle_points(figure=bottom, axis=1)

def get_bounds(fig, partition, lateral_wall) -> tuple[np.ndarray]:
    left_bound = partition[:, 0]

    if lateral_wall.shape[1] == 0:
        right_bound = fig[:, -1]
    else:
        right_bound = lateral_wall[:, -1]
    return (left_bound, right_bound)
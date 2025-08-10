import numpy as np
from typing import TypedDict
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from algorithms.average_width import find_middle_points
from algorithms.find_contour import find_contour_points
from core.figure import read_segmentation_file, get_frontal_slice, FrontalSlice, SegmentationImage

def unfold_spline(
        file_path: str, 
        front_diapason: tuple[int], 
        algorithm: str='middle_points'
    ) -> SegmentationImage:
    """Алгоритм развертки.

    1. Поиск одномерного представления фигуры среза (алгоритм middle_points или contour).
    2. Цикл по указанным срезам;
    3. Построение развертки каждого сегмента (перегородки, дна и боковой стенки);
    4. Объединение в общую развертку.

    ПРОБЛЕМЫ: 
        1. Если в середине среза нет дна носа, то алгоритм не работает;
        2. Если отсутствуют перегородка, дно или боковая стенка, так же не работает.

    Args: 
        - file_path - путь до сегментированного изображения (формат .nrrd)
        -  front_diapason - диапазон срезов по оси фронтальный срез (ось y)
        - algorithm - метод построения двумерного представления среза
            - middle_points - построение срединной линии
            - contour - построение контура

    Returns:
        unfold - словарь с разверткой срезов для каждого сегмента
    """
    if algorithm == 'middle_points':
        find_points_method = find_middle_points
    elif algorithm == 'contour':
        find_points_method = find_contour_points    
    else:
        raise ValueError(f"Алгоритм {algorithm} не поддерживается.\nИспользуйте middle_points или contour")

    segmentation_image = read_segmentation_file(file_path)

    # Цикл по указанным срезам
    for front_coord in range(front_diapason[0], front_diapason[1]):
        frontal_slice = get_frontal_slice(segmentation_image, front_coord)
    
        unfold_intervals = unfold_slice(frontal_slice, find_points_method)
        frontal_slice['unfold'] = unfold_intervals

        segmentation_image['slices'].append(frontal_slice)

    return segmentation_image

def unfold_slice(
        frontal_slice: FrontalSlice, 
        find_points_method,
    ) -> np.ndarray:
    """Построение развертки среза.

    1. Получаем список точек трех сегментов: перегородки, латеральной стенки и дна носа;
    2. Строим развертки каждого сегмента.
    
    Args:
        - frontal_slice - экземпляр фронтального среза;
        - find_points_method - метод поиска точек для построения развертки.
    
    Returns:
        - unfold_intervals - список интервалов разверток сегментов.
        
    """ 
    figure = frontal_slice['figure']

    partition, lateral_wall, bottom = find_points_method(figure)

    bottom_unfolded = bottom_unfold(bottom)

    partition_unfolded = partition_unfold(partition, bottom, bottom_unfolded)

    if lateral_wall.shape[1] > 2:
        lateral_wall_unfolded = lateral_wall_unfold(lateral_wall, bottom, bottom_unfolded)
    else:
        lateral_wall_unfolded = []
    
    

    unfold_intervals = np.array(partition_unfolded + bottom_unfolded + lateral_wall_unfolded)

    return unfold_intervals

def partition_unfold(
        partition: np.ndarray, 
        bottom: np.ndarray,
        bottom_unfolded: list[tuple[float]],
    ) -> list[tuple[float]]:
    """Построение развертки перегородки носа.
    
    1. Построение сплайна по точкам перегородки;
    2. Определение точек перфорации;
    3.1 Если перфорация найдена:
        3.1.1 Вычисляем длину сплайна до точки перфорации;
        3.1.2 Вычисляем длину перфорации;
        3.1.2 Вычисляем длину после перфорации;
    3.2 Если нет перфорации:
        3.2.1 Вычисляем длину перегородки


    Args:
        - lateral_wall - массив точек боковой стенки на срезе;
        - bottom - массив точек дна носа на срезе;
    Returns:
        - partition_unfold - список интервалов развертки перегородки
    """
    spline = calculate_spline(partition, axis=0)
    perforation_bounds = find_perforation_bounds(partition)

    start = partition[:, -1] # верхняя точка перегородки
    end = bottom[:, 0] # левая граница дна носа

    partition_unfold: list[tuple[float, float]] = []
    if perforation_bounds is not None:
        start_perf, end_perf = perforation_bounds[0], perforation_bounds[1] 

        # Расчет идет в обратном направлении от начала дна носа до начала перегородки.
        length_before_perf = arc_length(end_perf[0], end[0], spline) # Расстояние от конца перегородки до начала перфорации
        end_perforation_point = end[1] - length_before_perf # Точка конца перфорации


        length_perf = arc_length(start_perf[0], end_perf[0], spline) # Расстояние перфорации
        start_perforation_point = end_perforation_point - length_perf # Точка начала перфорации

        length_after_perf = arc_length(start[0], start_perf[0], spline) # Расстояние до начала перегородки
        start_partition_point = start_perforation_point - length_after_perf # Точка начала перегородки

        end_partition_point = bottom_unfolded[0][0] # Точка конца перегородки находится в точке начала дна носа

        partition_unfold.append((start_partition_point, start_perforation_point))
        partition_unfold.append((end_perforation_point, end_partition_point))

    else:
        length = arc_length(start[0], end[0], spline)
        start_partition_point = end[1] - length
        end_partition_point = bottom_unfolded[0][0]

        partition_unfold.append((start_partition_point, end_partition_point))
        
    return partition_unfold

def calculate_spline(figure: np.ndarray, axis:int) -> CubicSpline:
    """Вычисление сплайна фигуры.

    1. Определяет точки для формирования сплайна:
        Выбираются три точки: начальная, средняя и последняя;
    2. Вычисляет сплайн;
    
    Args:
        figure - массив точек фигуры;
        axis - ось, по которой вычисляется сплайн. по x - 1, по z - 0.
    Для перегородки и латеральной стенки axis = 0, для дна носа axis = 1
    """

    points = np.array([figure[:, 0], figure[:, figure.shape[1] // 2], figure[:, -1]]).T # получаем начальную, среднюю и последнюю точки фигуры
    
    spline = CubicSpline(points[axis], points[axis ^ 1])
        
    return spline

def arc_length(start, end, spline) -> float:
    """Поиск длины дуги от начальной точки до конечной"""

    integrand = lambda x: np.sqrt(1 + (spline.derivative()(x))**2)
    length, _ = quad(integrand, start, end)
    return abs(length)

def find_perforation_bounds(partition: np.ndarray, threshold: int=10) -> tuple[np.ndarray]:
    """Поиск точек начала и конца перфорации.
    
    Args:
        partition - массив точек перфорации;
        threshold - расстояние между точками.

    Returns:
        кортеж из двух точек: начало и конец перфорации.
    """
    z = partition[0]
    if len(z) < 2:
        return None  
    
    max_diff = 0
    idx = 0
    
    for i in range(len(z) - 1):
        diff = abs(z[i+1] - z[i])
        if diff > max_diff:
            max_diff = diff
            idx = i  

    if max_diff < threshold:
        return None
    
    return (partition[:, idx], partition[:, idx+1])

def lateral_wall_unfold(
        lateral_wall: np.ndarray, 
        bottom: np.ndarray,
        bottom_unfolded: list[tuple[float]], 
    ) -> list[tuple[float]]:
    """Построение развертки боковой стенки носа.
    
    1. Построение сплайна по точкам боковой стенки;
    2. Вычисление длины построенного сплайна.

    Args:
        - lateral_wall - массив точек боковой стенки на срезе;
        - bottom - массив точек дна носа на срезе;
    
    Returns:
        - lateral_wall_unfold - интервал развертки боковой стенки

    """
    spline = calculate_spline(lateral_wall, axis=0)
    start = bottom[:, -1]
    end = lateral_wall[:, -1]
    length = arc_length(start[0], end[0], spline)

    start_lateral_wall_point = bottom_unfolded[0][1] 
    end_lateral_wall_point = start[1] + length
    lateral_wall_unfold: list[tuple[float]] = [(start_lateral_wall_point, end_lateral_wall_point)]
    
    return lateral_wall_unfold

def bottom_unfold(
        bottom: np.ndarray, 
    ) -> list[tuple[float]]:
    """Развертка дна носа.
    
    1. Расчет сплайна по точкам дна носа;
    2. Вычисление длины построенного сплайна.
    
    Args:
        - bottom - массив точек дна носа на срезе;
        - axial_coord - координата аксиального среза.

    Returns:
        - bottom_unfolded - интервал развертки дна носа
    """
    spline = calculate_spline(bottom, axis=1)
    start = bottom[:, 0]
    end = bottom[:, -1]
    length = arc_length(start[1], end[1], spline)

    end_bottom_point = start[1] + length
    bottom_unfolded: list[tuple[float]] = [(start[1], end_bottom_point)]
    
    return bottom_unfolded




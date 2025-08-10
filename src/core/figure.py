import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import TypedDict, Optional


class FrontalSlice(TypedDict):
    id: int # Индекс фронтального среза
    image_2D: np.ndarray # Пиксели фронтального среза
    figure: np.ndarray # Бинарное изображение среза (1 - это фигура)
    unfold_intervals: np.ndarray # Массив точек развертки

class SegmentationImage(TypedDict):
    name: str
    image_3D: np.ndarray # Трехмерное изображение (z, y, x)
    spacing: tuple[float] # Дистанция между пикселями по направлениям в мм (x, y, z)
    direction_matrix: tuple[float] # Матрица косинусов направления
    direction: np.ndarray[int]
    slices: list[FrontalSlice] # Список срезов сегментированного изображения

def read_segmentation_file(file_path:str) -> SegmentationImage:
    """Функция чтения файла сегментации. 
    Файл формата .nrrd.

    Returns:
        segmentation_image -  объект сегментированного изображения, который содержит:
            - image_3D - трехмерный массив пикселей сегментированного изображения
            - spacing - дистанция между пикселями по каждому измерению в миллиметрах
            - direction_matrix - матрица косинусов направления
            - direction - направления осей (z, x) - то есть сагиттального и аксиального соответственно.
    """
    image = sitk.ReadImage(file_path)

    image_3D = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    direction_matrix = image.GetDirection()
    direction = np.array([direction_matrix[8], direction_matrix[0]]) # Направления главных осей (z, x)

    return SegmentationImage(
        name=Path(file_path).stem,
        image_3D=image_3D,
        spacing=spacing,
        direction_matrix=direction_matrix,
        direction=direction,
        slices=[]
    )

def get_frontal_slice(
        segmentation_image: SegmentationImage,
        front_coord: int, 
    ) -> Optional[FrontalSlice]:
    """Из трехмерного изображения получает двумерный изображение и фигуру срезаю.

    Фигура - это все не нулевые значения пикселей среза.
    
    Args:
        - image_3D: трехмерный массив пикселей сегментированного изображения;
        - front_coord: координата фронтального среза;
        - direction_matrix: матрица направления.
    Returns:
        FrontalSlice - фронтальный срез:
            - image_2D - двумерный массив пикселей фронтального среза
            - figure - бинарный массив фигуры среза
        
    """
    image_3D  = segmentation_image['image_3D']
    direction = segmentation_image['direction']

    image_2D = image_3D[:, front_coord, :].copy()
    figure = np.asarray(image_2D.nonzero())
    
    if figure.shape[1] == 0: # Если пустой срез - пропускаем
        return None

    image_2D, figure = crop_slice_image_and_figure(image_2D, figure)

    for i in range(2):
        figure[i] = figure[i] * direction[i]
        figure[i] += np.abs(np.min(figure[i]))

    return FrontalSlice(
        id=front_coord,
        image_2D=image_2D,
        figure=figure
    )

def crop_slice_image_and_figure(slice_image, fig):
    """Масштабирование среза по границам фигуры"""
    
    minx, maxx, minz, maxz = get_boundary_of_figure(fig)

    cropped_slice_image = slice_image[minz:maxz, minx:maxx]

    fig = np.asarray(cropped_slice_image.nonzero())
    fig = fig[:, np.lexsort((fig[0], fig[1]))]
    fig[0], fig[1] = fig[0], fig[1]

    return cropped_slice_image, fig

def get_boundary_of_figure(fig):
    """Получение границ фигуры"""
    z = fig[0]
    x = fig[1]
    return min(x), max(x), min(z), max(z)

def get_center_of_figure(fig):
    """Найти максимальную координату по z центра фигуры"""
    median_x = (max(fig[1]) - min(fig[1])) // 2
    median_z = (max(fig[0]) - min(fig[0])) // 2

    mask = (fig[1] >= median_x - 1) & (fig[1] <= median_x + 1) & (fig[0] < median_z * 0.5)
    
    return fig[:, mask][:, -1]

def find_left_boundary(fig):
    """
    Найти левую границу фигуры 
    (максимальное значение по z слева от центра)
    """

    z = max(fig[0])
    mask = fig[0] == z
    x = min(fig[1, mask])
    return (z, x)

def find_right_boundary(fig, center):
    """
    Найти правую границу фигуры 
    (максимальное значение по z справа от центра)
    """
    return find_left_boundary(fig[:, fig[1] > center])


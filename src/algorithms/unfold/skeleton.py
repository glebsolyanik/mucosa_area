import numpy as np
from typing import TypedDict
from skan import csr
from scipy.ndimage import center_of_mass
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk, opening, skeletonize
from skimage.filters import gaussian
from scipy.ndimage import distance_transform_edt
from dsepruning import skel_pruning_DSE

from core.figure import read_segmentation_file, get_frontal_slice, FrontalSlice, SegmentationImage

class SkeletonInfo(TypedDict):
    id: int
    length: float
    coords: np.ndarray
    distance_to_sibling: float

def unfold_skeleton(file_path: str, front_diapason: tuple[int]) -> SegmentationImage:
    """Алгоритм развертки сегментированного изображения с помощью метода скелета.

    1. Формирование двумерных срезов сегментированного изображения
    2. Развертка каждого среза
    
    """
    segmentation_image = read_segmentation_file(file_path)

    if front_diapason is None:
        front_diapason = (0, segmentation_image['image_3D'].shape[1])

    for front_coord in range(front_diapason[0], front_diapason[1]):
        frontal_slice = get_frontal_slice(segmentation_image, front_coord)

        if frontal_slice:
            unfold = unfold_slice(frontal_slice, segmentation_image['spacing'])
            frontal_slice['unfold'] = unfold

            segmentation_image['slices'].append(frontal_slice)

    return segmentation_image


def unfold_slice(frontal_slice: FrontalSlice, spacing: tuple[float]) -> np.ndarray:
    """Построение развертки среза.

    Используется метод скелетизации и библиотека анализа скелета skan.
    
    1. Подготавливаем изображение среза;
    2. Получаем скелет(ы) изображения, обрезаем ненужные ветки;
    3. Получаем индексы найденных скелетов;
    4. Для каждого скелета определяем длину основной ветки;
    5. Рассчитываем расстояние между скелетами;
    6. Строим точки относительно центра масс

    Args:
        frontal_slice - фронтальный срез сегментированного изображения
        spacing - ...

    Returns:
        - unfold_points - массив точек развертки
    """
    image_2D = frontal_slice['image_2D']

    image_2D = _prepare_slice(image_2D)

    skeleton = skeletonize(image_2D[::-1, :], method='lee')
    dist = distance_transform_edt(image_2D[::-1, :], return_indices=False, return_distances=True)
    skeleton = skel_pruning_DSE(skeleton, dist, 90)

    skeleton_skan = csr.Skeleton(skeleton, spacing=np.array((spacing[2], spacing[0])))
    skeleton_skan_data = csr.summarize(skeleton_skan, separator='_')

    # Получаем индексы скелетов (может быть несколько из-за перфорации)
    skeletons: list[SkeletonInfo] = [SkeletonInfo(id=idx) for idx in skeleton_skan_data['skeleton_id'].unique()]
    # Получаем длину каждого скелета
    for skeleton in skeletons:
        skeleton_data = skeleton_skan_data[skeleton_skan_data['skeleton_id'] == skeleton['id']]
        skeleton['length'] = sum(skeleton_data['branch_distance'])

        # Получаем координаты веток скелета
        branch_indexes = skeleton_data.index
        skeleton_coords = []
        for branch_idx in branch_indexes:
            skeleton_coords += skeleton_skan.path_coordinates(branch_idx).tolist()
        skeleton['coords'] = np.array(skeleton_coords)

    if len(skeletons) > 1: 
        skeletons = _find_distance_between_skeletons(skeletons, spacing)

    unfold_points:list[int] = [0]
    for skeleton in skeletons:
        last_point = unfold_points[-1]
        new_point = last_point + skeleton['length']
        unfold_points.append(new_point)

        if 'distance_to_sibling' in skeleton.keys():
            distance_point = new_point + skeleton['distance_to_sibling']
            unfold_points.append(distance_point)
    
    slice_center_mass = center_of_mass(image_2D)
    unfold_points = np.array(unfold_points) - slice_center_mass[1] * spacing[2]

    reshaped = unfold_points.reshape(-1, 2)
    unfold_intervals = np.array([tuple(row) for row in reshaped])

    return unfold_intervals

def _prepare_slice(image_2D: np.ndarray):
    """Подготовка изображения скелета.
    
    1. Открытие - удаляем мелкие шумы, сглаживаем контур;
    2. Закрытие - заполняем пустое пространство объектов;
    3. Удаление маленьких объектов;
    4. Удаляем маленькие дырки;
    5. Фильтр Гаусса - сглаживаем резкие шумы;
    6. Преобразуем в бинарное изображение.

    Args:
        image_2D - двумерный срез
    
    Returns:
        image_2D - обработанный двумерный срез
    
    """
    image_2D = opening(image_2D, disk(2))
    image_2D = closing(image_2D, disk(3))
    image_2D = remove_small_objects(image_2D, min_size=100)
    image_2D = remove_small_holes(image_2D, area_threshold=100)
    image_2D = gaussian(image_2D, sigma=1)

    image_2D = image_2D > 0.6 # можно попробовать другое значение

    return image_2D

def _find_distance_between_skeletons(skeletons: list[SkeletonInfo], spacing: tuple[float]):
    """Расчет расстояния между двумя скелетами."""
    for i in range(len(skeletons) - 1):
        s_coords_1 = skeletons[i]['coords']
        s_coords_2 = skeletons[i + 1]['coords']

        s_coords_1 = s_coords_1[np.argsort(s_coords_1[:, 0])]
        s_coords_2 = s_coords_2[np.argsort(s_coords_2[:, 1])]

        skeletons[i]['distance_to_sibling'] = \
            np.sqrt (np.sum((s_coords_1[-1] - s_coords_2[0])**2)) * spacing[0]
        
    return skeletons




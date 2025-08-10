import vtk

from core.figure import SegmentationImage

def save_unfold_to_vtk(segmentation_image: SegmentationImage):
    points = vtk.vtkPoints()
    vtk_lines = vtk.vtkCellArray()
    
    def _add_line(line_points, front_coord):
        ids = []
        for pt in line_points:
            pt_3D = (0, front_coord, pt) # Трехмерная точка
            ids.append(points.InsertNextPoint(pt_3D))

        vtk_line = vtk.vtkLine()
        vtk_line.GetPointIds().SetId(0, ids[0])
        vtk_line.GetPointIds().SetId(1, ids[1])
        vtk_lines.InsertNextCell(vtk_line)

    slices = segmentation_image['slices']

    for front_slice in slices:
        unfold_intervals = front_slice['unfold']
        idx = front_slice['id']

        for interval in unfold_intervals:
            _add_line(interval, idx)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(vtk_lines) 

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(f"{segmentation_image['name']}_unfold.vtp")
    writer.SetInputData(polyData)
    writer.Write() 

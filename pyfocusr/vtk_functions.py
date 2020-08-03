import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def read_vtk_mesh(path_to_file):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path_to_file)
    reader.Update()
    return reader.GetOutput()


def icp_transform(target, source, numberOfIterations=100, number_landmarks=1000, transform_mode='rigid'):
    icp = vtk.vtkIterativeClosestPointTransform()
    if transform_mode == 'rigid':
        icp.GetLandmarkTransform().SetModeToRigidBody()
    elif transform_mode == 'similarity':
        icp.GetLandmarkTransform().SetModeToSimilarity()
    else:
        raise('Error invalid transform mode')
    icp.SetTarget(target)
    icp.SetSource(source)
    icp.SetMaximumNumberOfIterations(numberOfIterations)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    icp.SetMaximumNumberOfLandmarks(number_landmarks)
    return icp


def apply_transform(source, transform):
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(source)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()


def get_node_curvatures(vtk_mesh, curvature_type='min'):
    curvature = vtk.vtkCurvatures()
    if curvature_type == 'min':
        curvature.SetCurvatureTypeToMinimum()
    elif curvature_type == 'max':
        curvature.SetCurvatureTypeToMaximum()
    curvature.SetInputData(vtk_mesh)
    curvature.Update()
    return curvature.GetOutput()


def get_max_curvature(vtk_mesh):
    return [vtk_to_numpy(get_node_curvatures(vtk_mesh, curvature_type='max').GetPointData().GetScalars()),]


def get_min_curvature(vtk_mesh):
    return [vtk_to_numpy(get_node_curvatures(vtk_mesh, curvature_type='min').GetPointData().GetScalars()),]


def get_min_max_curvature_values(vtk_mesh):
    min_curvatures = get_node_curvatures(vtk_mesh, curvature_type='min')
    max_curvatures = get_node_curvatures(vtk_mesh, curvature_type='max')

    min_curvature_values = vtk_to_numpy(min_curvatures.GetPointData().GetScalars())
    max_curvature_values = vtk_to_numpy(max_curvatures.GetPointData().GetScalars())

    return min_curvature_values, max_curvature_values


def vtk_deep_copy(mesh):
    new_mesh = vtk.vtkPolyData()
    new_mesh.DeepCopy(mesh)

    return new_mesh

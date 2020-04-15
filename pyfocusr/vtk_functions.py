import vtk


def icp_transform(target, source, numberOfIterations=100, number_landmarks=1000):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetTarget(target)
    icp.SetSource(source)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(numberOfIterations)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    icp.SetMaximumNumberOfLandmarks(number_landmarks)
    return icp


def apply_icp_transform(source, icp):
    icp_transform_filter = vtk.vtkTransformPolyDataFilter()
    icp_transform_filter.SetInputData(source)
    icp_transform_filter.SetTransform(icp)
    icp_transform_filter.Update()
    return icp_transform_filter.GetOutput()

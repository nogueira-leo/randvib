import numpy as np
from vtk import vtkUnstructuredGrid, vtkPoints, vtkDoubleArray, vtkXMLUnstructuredGridWriter

def vtk_write(U, df0val, data, coord, connect, nnode, nel):

# Gera VTU da Malha ###################################################
    my_vtk_dataset = vtkUnstructuredGrid()
    points = vtkPoints()
    VTK_QUAD = 9
    for id in range(nnode):
        points.InsertPoint(id, [coord[id,0], coord[id,1], coord[id,2]])
        my_vtk_dataset.SetPoints(points)
    my_vtk_dataset.Allocate(nel)
    for id in range(nel):
        point_ids = [connect[id,0]-1, connect[id,1]-1, connect[id,2]-1, connect[id,3]-1]
        my_vtk_dataset.InsertNextCell(VTK_QUAD, 4, point_ids)

    unod1 = np.zeros((nnode,3), dtype=complex)
    eigm1 = np.zeros((nnode,3), dtype=float)
    
    for i in range(nnode):
        unod1[i,0] = U[5*i]
        unod1[i,1] = U[5*i+1]
        unod1[i,2] = U[5*i+2]
        #
        #eigm1[i,0] = modal_shape[5*i]
        #eigm1[i,1] = modal_shape[5*i+1]
        #eigm1[i,2] = modal_shape[5*i+2]
                        
    array1 = vtkDoubleArray()
    array1.SetNumberOfComponents(3)
    array1.SetNumberOfTuples(nnode)
    array1.SetName('Displacement - Real')
    #
    #array2 = vtkDoubleArray()
    #array2.SetNumberOfComponents(3)
    #array2.SetNumberOfTuples(nnode)
    #array2.SetName('Eigenvector')

    for id in range(nnode):
        values1 = [np.real(unod1[id,0]), np.real(unod1[id,1]), np.real(unod1[id,2])]
        array1.SetTuple(id, values1)
        my_vtk_dataset.GetPointData().AddArray(array1)
        #
        #values2 = [(eigm1[id,0]), (eigm1[id,1]), (eigm1[id,2])]
        #array2.SetTuple(id, values2)
        #my_vtk_dataset.GetPointData().AddArray(array2)
        #
    
    array3 = vtkDoubleArray()
    array3.SetNumberOfComponents(1)
    array3.SetNumberOfTuples(nel)
    array3.SetName('Espessura')

    array4 = vtkDoubleArray()
    array4.SetNumberOfComponents(1)
    array4.SetNumberOfTuples(nel)
    array4.SetName('Derivada da compliance')

    for i in range(nel):
        array3.SetTuple(i, [np.real(data[i])])
        my_vtk_dataset.GetCellData().AddArray(array3)
        array4.SetTuple(i, [np.real(df0val[i])])
        my_vtk_dataset.GetCellData().AddArray(array4)
        
    # Write file
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName("result.vtu")
    writer.SetInputData(my_vtk_dataset)
    writer.Write()
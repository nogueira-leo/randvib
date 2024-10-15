import numpy as np
from vtk import vtkUnstructuredGrid, vtkPoints, vtkDoubleArray, vtkXMLUnstructuredGridWriter

def vtk_write_displacement(U, freq, coord, connect, nnode, nel):

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

    
    
    
    for ff in range(len(freq)):
        unod1 = np.zeros((nnode,3), dtype=complex)
        
        for i in range(nnode):
            unod1[i,0] = U[5*i,ff]
            unod1[i,1] = U[5*i+1,ff]
            unod1[i,2] = U[5*i+2,ff]
        
        array1 = vtkDoubleArray()
        array1.SetNumberOfComponents(3)
        array1.SetNumberOfTuples(nnode)
        array1.SetName(f'Displacement {ff+1:03d} ({freq[ff]:.2f} Hz) - Real')

        for id in range(nnode):
            values1 = [np.real(unod1[id,0]), np.real(unod1[id,1]), np.real(unod1[id,2])]
            array1.SetTuple(id, values1)
            my_vtk_dataset.GetPointData().AddArray(array1)

    
                        
    
    

    
 
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName("displacements.vtu")
    writer.SetInputData(my_vtk_dataset)
    writer.Write()


def vtk_write_velocity(V, freq, coord, connect, nnode, nel):

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

    
    
    
    for ff in range(len(freq)):
        unod1 = np.zeros((nnode,3), dtype=complex)
        
        for i in range(nnode):
            unod1[i,0] = V[5*i,ff]
            unod1[i,1] = V[5*i+1,ff]
            unod1[i,2] = V[5*i+2,ff]
        
        array1 = vtkDoubleArray()
        array1.SetNumberOfComponents(3)
        array1.SetNumberOfTuples(nnode)
        array1.SetName(f'Velocity {ff+1:03d} ({freq[ff]:.2f} Hz) - Real')

        for id in range(nnode):
            values1 = [np.real(unod1[id,0]), np.real(unod1[id,1]), np.real(unod1[id,2])]
            array1.SetTuple(id, values1)
            my_vtk_dataset.GetPointData().AddArray(array1)

    
                        
    
    

    
 
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName("velocity.vtu")
    writer.SetInputData(my_vtk_dataset)
    writer.Write()



def vtk_write_modal(modal_shape, modal_freq, coord, connect, nnode, nel):
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

    eigm1 = np.zeros((nnode,3), dtype=float)

    for ff in range(len(modal_freq)):
        for i in range(nnode):
            eigm1[i,0] = modal_shape[5*i,ff]
            eigm1[i,1] = modal_shape[5*i+1,ff]
            eigm1[i,2] = modal_shape[5*i+2,ff]

        array2 = vtkDoubleArray()
        array2.SetNumberOfComponents(3)
        array2.SetNumberOfTuples(nnode)
        array2.SetName(f'Eigenvector {ff+1:03d} - ({modal_freq[ff]:.2f})')

        for id in range(nnode):
        
            values2 = [(eigm1[id,0]), (eigm1[id,1]), (eigm1[id,2])]
            array2.SetTuple(id, values2)
            my_vtk_dataset.GetPointData().AddArray(array2)
        
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName("modal.vtu")
    writer.SetInputData(my_vtk_dataset)
    writer.Write()
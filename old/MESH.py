import gmsh
import numpy as np
import vtk
import math

def malha2D_QUAD4(lx,ly,size):

    # Inicializar o Gmsh
    gmsh.initialize()

    # Criar ou carregar a geometria
    gmsh.model.add("Quadrangular Mesh Example")
    # Aqui você pode criar uma geometria, por exemplo, um retângulo
    # Para um arquivo IGES, use: gmsh.open("sua_geometria.iges")

    lc = size  # Tamanho do elemento da malha
    # Criar um retângulo como exemplo
    #p1 = gmsh.model.geo.addPoint(0,  0,  0, lc)
    #p2 = gmsh.model.geo.addPoint(lx, 0,  0, lc)
    #p3 = gmsh.model.geo.addPoint(lx, ly, 0, lc)
    #p4 = gmsh.model.geo.addPoint(0,  ly, 0, lc)

    p1 = gmsh.model.geo.addPoint(-lx/2,  -ly/2,  0, lc)
    p2 = gmsh.model.geo.addPoint( lx/2,  -ly/2,  0, lc)
    p3 = gmsh.model.geo.addPoint( lx/2,   ly/2,  0, lc)
    p4 = gmsh.model.geo.addPoint(-lx/2,   ly/2,  0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Criar uma curva fechada e uma superfície plana
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    # Aplicar malha transfinita à superfície
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)  # Recombinar para obter quadriláteros

   # Sincronizar a geometria
    gmsh.model.geo.synchronize()

    # Gerar a malha bidimensional
    gmsh.model.mesh.generate(2)

    # Recompor para gerar elementos quadrangulares
    gmsh.model.mesh.recombine()

    # Sincronizar novamente para garantir que todas as operações foram aplicadas
    gmsh.model.geo.synchronize()

    # Obter os nós da malha
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)

    # Verifique as coordenadas dos nós
    # print("Coordenadas dos nós:")
    # print(node_coords)
    gmsh.fltk.run()

    # Obter conectividade dos elementos quadrangulares
    element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements(dim=2)

    # Procurar elementos quadrangulares (element_type 3)
    quad_elements = []
    for i, element_type in enumerate(element_types):
        if element_type == 3:  # 3 corresponde a elementos quadrangulares de 4 nós
            quad_elements = np.array(node_tags_per_element[i], dtype=int).reshape(-1, 4)

    # Verifique se os elementos quadrangulares foram encontrados
    # print("Conectividades dos elementos quadrangulares:")
    # print(quad_elements)

    # Finalizar a API do Gmsh
    gmsh.finalize()

    # Exibir informações sobre a malha quadrangular gerada
    # print("Nós da malha:\n", node_coords)
    # print("Conectividades dos elementos quadrangulares:\n", quad_elements)

    # Filtrar nós que estão dentro dos limites especificados
    nodes_in_faceX1 = np.where(
        (node_coords[:, 0] >= (-lx/2)-(size/8)) & (node_coords[:, 0] <= (-lx/2)+(size/8))  # Dentro do intervalo x
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

    nodes_in_faceX2 = np.where(
        (node_coords[:, 0] >= lx/2-(size/8)) & (node_coords[:, 0] <= lx/2+(size/8))  # Dentro do intervalo x
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

      # Filtrar nós que estão dentro dos limites especificados
    nodes_in_faceY1 = np.where(
        (node_coords[:, 1] >= -ly/2-(size/8)) & (node_coords[:, 1] <= -ly/2+(size/8))  # Dentro do intervalo y
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

    nodes_in_faceY2 = np.where(
        (node_coords[:, 1] >= ly/2-(size/8)) & (node_coords[:, 1] <= ly/2+(size/8))  # Dentro do intervalo y
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

    middle_line = np.where(
        (node_coords[:, 1] >= 0-(size/8)) & (node_coords[:, 1] <= 0+(size/8))   # Dentro do intervalo y
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices
    #nodes_under_load = np.where(
    #middle_node = np.where(
    #    (node_coords[:, 1] >= (ly/2)-(size/8)) & (node_coords[:, 1] <= (ly/2)+(size/8)) & (node_coords[:, 0] >= (lx/2)-(size/8)) & (node_coords[:, 0] <= (lx/2)+(size/8) )  # Dentro do intervalo y
    #)[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices
    #nodes_under_load = np.where(
    #    (node_coords[:, 0] >= 0+(size/8)) & (node_coords[:, 0] >= lx-(size/8)) &
    #    (node_coords[:, 1] >= 0+(size/8)) & (node_coords[:, 1] >= ly-(size/8)) 
    #)

    return node_coords, quad_elements, nodes_in_faceX1, nodes_in_faceX2, nodes_in_faceY1, nodes_in_faceY2, middle_line
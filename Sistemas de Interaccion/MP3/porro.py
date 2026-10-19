"""
Script Procedural para Blender (bpy + bmesh)
Modelo: Cigarrillo Herbal Artesanal Low-Poly (Estilo Pre-Roll)
Propósito: Activo 3D optimizado para entornos de videojuegos con fines educativos.

Características técnicas:
- Topología limpia generada desde cero con BMesh (sin caras internas ni geometría residual).
- Circunferencia base de 12 segmentos radiales.
- Cuerpo cónico con 4 subdivisiones longitudinales y ligera deformación orgánica.
- Filtro cilíndrico con boquilla ahuecada en la base.
- Punta torcida y plegada de bajo poligonaje con triangulación facetada.
- Sombreado plano (Flat Shading) y materiales PBR simples sin texturas pesadas.
- Conteo geométrico: ~170 vértices topológicos (~336 triángulos / ~500 vértices de GPU en Flat Shading).
"""

import bpy
import bmesh
import math
import random
from mathutils import Vector, Matrix


def limpiar_objeto_previo(nombre_objeto: str):
    """Elimina versiones previas del objeto y su malla para evitar duplicados en escena."""
    if nombre_objeto in bpy.data.objects:
        obj_previo = bpy.data.objects[nombre_objeto]
        malla_previa = obj_previo.data
        bpy.data.objects.remove(obj_previo, do_unlink=True)
        if malla_previa and malla_previa.users == 0:
            bpy.data.meshes.remove(malla_previa)


def crear_material_simple(nombre: str, color_rgba: tuple, rugosidad: float) -> bpy.types.Material:
    """Crea o actualiza un material simple basado en Principled BSDF sin texturas."""
    mat = bpy.data.materials.get(nombre)
    if mat is None:
        mat = bpy.data.materials.new(name=nombre)

    mat.use_nodes = True
    nodos = mat.node_tree.nodes
    bsdf = nodos.get("Principled BSDF")

    if bsdf:
        bsdf.inputs["Base Color"].default_value = color_rgba
        bsdf.inputs["Roughness"].default_value = rugosidad
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.2
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.2

    # Color en viewport para previsualización en modo Solid
    mat.diffuse_color = color_rgba
    mat.roughness = rugosidad
    return mat


def generar_preroll_lowpoly():
    # Semilla fija para garantizar deformaciones orgánicas reproducibles
    rng = random.Random(42)

    nombre_activo = "PreRoll_Herbal_LowPoly"
    limpiar_objeto_previo(nombre_activo)

    # =========================================================================
    # 1. PARÁMETROS DE GEOMETRÍA (LOW-POLY)
    # =========================================================================
    SEGMENTOS_RADIALES = 12  # Entre 8 y 12 segmentos según especificación

    # Definición de anillos longitudinales:
    # (altura_z, radio, torsion_grados, intensidad_ruido, indice_material, triangular_seccion)
    # Material 0 = Cartón (Filtro) | Material 1 = Papel (Cuerpo y Punta)
    perfiles = [
        # --- Interior ahuecado de la boquilla (Filtro) ---
        (0.035, 0.028,   0.0, 0.0000, 0, False),  # Fondo interno del filtro
        (0.000, 0.028,   0.0, 0.0000, 0, False),  # Borde interior inferior
        (0.000, 0.038,   0.0, 0.0000, 0, False),  # Borde exterior inferior (base del filtro)

        # --- Exterior del Filtro (3 segmentos longitudinales) ---
        (0.085, 0.040,   0.0, 0.0002, 0, False),
        (0.170, 0.042,   0.0, 0.0002, 0, False),
        (0.255, 0.045,   0.0, 0.0004, 0, False),  # Transición filtro -> papel

        # --- Cuerpo Cónico de Papel (4 subdivisiones longitudinales) ---
        (0.415, 0.056,   2.0, 0.0018, 1, True),
        (0.575, 0.068,   4.5, 0.0022, 1, True),
        (0.735, 0.081,   7.0, 0.0026, 1, True),
        (0.895, 0.094,  10.0, 0.0030, 1, True),  # Parte más ancha del cono

        # --- Cierre y Punta Torcida (bajo poligonaje) ---
        (0.945, 0.068,  22.0, 0.0025, 1, True),  # Hombro del cierre
        (0.985, 0.020,  55.0, 0.0015, 1, True),  # Cuello estrangulado (torsión)
        (1.030, 0.046,  90.0, 0.0040, 1, True),  # Expansión del papel doblado
        (1.080, 0.052, 120.0, 0.0045, 1, True),  # Corona rústica de la punta
        (1.115, 0.028, 145.0, 0.0030, 1, True),  # Cierre superior
    ]

    # =========================================================================
    # 2. CONSTRUCCIÓN DE LA MALLA CON BMESH
    # =========================================================================
    bm = bmesh.new()
    anillos_vertices = []

    for idx_anillo, (z, radio, torsion_deg, ruido, mat_idx, _) in enumerate(perfiles):
        anillo_actual = []
        torsion_rad = math.radians(torsion_deg)

        # Ligerísimo desplazamiento axial en el cuerpo/punta para romper la simetría perfecta
        offset_x = rng.uniform(-ruido, ruido) * 0.8 if z > 0.25 else 0.0
        offset_y = rng.uniform(-ruido, ruido) * 0.8 if z > 0.25 else 0.0

        for i in range(SEGMENTOS_RADIALES):
            angulo = (2.0 * math.pi * i / SEGMENTOS_RADIALES) + torsion_rad

            # Variación radial y vertical controlada para dar aspecto artesanal
            var_r = radio + rng.uniform(-ruido, ruido)
            var_z = z + (rng.uniform(-ruido * 0.6, ruido * 0.6) if z > 0.26 else 0.0)

            x = math.cos(angulo) * var_r + offset_x
            y = math.sin(angulo) * var_r + offset_y

            v = bm.verts.new((x, y, var_z))
            anillo_actual.append(v)

        anillos_vertices.append(anillo_actual)

    bm.verts.ensure_lookup_table()

    # Crear tapa interna en el fondo del hueco del filtro (sin vértices innecesarios)
    cara_fondo_filtro = bm.faces.new(anillos_vertices[0])
    cara_fondo_filtro.material_index = 0
    cara_fondo_filtro.smooth = False

    # Conectar anillos consecutivos mediante caras (puenteo longitudinal)
    caras_a_triangular = []

    for idx in range(len(perfiles) - 1):
        anillo_a = anillos_vertices[idx]
        anillo_b = anillos_vertices[idx + 1]
        mat_idx_segmento = perfiles[idx + 1][4]
        triangular_segmento = perfiles[idx + 1][5]

        # Caso especial: transición de filtro a papel (el segmento entre índice 4 y 5 es filtro)
        if idx == 4:
            mat_idx_segmento = 0
            triangular_segmento = False

        for i in range(SEGMENTOS_RADIALES):
            i_sig = (i + 1) % SEGMENTOS_RADIALES
            v1 = anillo_a[i]
            v2 = anillo_a[i_sig]
            v3 = anillo_b[i_sig]
            v4 = anillo_b[i]

            cara = bm.faces.new((v1, v2, v3, v4))
            cara.material_index = mat_idx_segmento
            cara.smooth = False

            if triangular_segmento:
                caras_a_triangular.append(cara)

    # Cierre del remate superior (vértice ápice de la punta doblada)
    vert_punta = bm.verts.new((0.004, -0.003, 1.132))
    ultimo_anillo = anillos_vertices[-1]
    for i in range(SEGMENTOS_RADIALES):
        i_sig = (i + 1) % SEGMENTOS_RADIALES
        cara_punta = bm.faces.new((ultimo_anillo[i], ultimo_anillo[i_sig], vert_punta))
        cara_punta.material_index = 1
        cara_punta.smooth = False

    # Triangulación selectiva en el papel y punta para resaltar el facetado low-poly
    if caras_a_triangular:
        res_tri = bmesh.ops.triangulate(
            bm,
            faces=caras_a_triangular,
            quad_method='FIXED',
            ngon_method='BEAUTY'
        )
        for f in res_tri.get("faces", []):
            f.smooth = False

    # Recalcular normales hacia el exterior y asegurar Flat Shading en toda la malla
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = False

    # =========================================================================
    # 3. CREACIÓN DEL OBJETO Y ASIGNACIÓN DE MATERIALES
    # =========================================================================
    malla = bpy.data.meshes.new(nombre_activo + "_Mesh")
    bm.to_mesh(malla)
    bm.free()

    # Garantizar sombreado plano (compatible con Blender 3.x y 4.x)
    for poly in malla.polygons:
        if hasattr(poly, "use_smooth"):
            poly.use_smooth = False

    objeto = bpy.data.objects.new(nombre_activo, malla)
    bpy.context.collection.objects.link(objeto)

    # Crear y asignar los 2 materiales simples requeridos
    mat_carton = crear_material_simple(
        nombre="Mat_Filtro_Carton",
        color_rgba=(0.76, 0.57, 0.36, 1.0),  # Marrón claro / beige cartón
        rugosidad=0.88
    )
    mat_papel = crear_material_simple(
        nombre="Mat_Papel_Herbal",
        color_rgba=(0.93, 0.90, 0.82, 1.0),  # Blanco hueso / crema
        rugosidad=0.78
    )

    objeto.data.materials.append(mat_carton)  # Slot 0
    objeto.data.materials.append(mat_papel)   # Slot 1

    # Posicionar e inclinar ligeramente como en la vista de presentación
    objeto.rotation_euler = (math.radians(65.0), 0.0, math.radians(-50.0))
    objeto.location = (0.0, 0.0, 0.04)

    # Seleccionar y activar el objeto generado
    bpy.ops.object.select_all(action='DESELECT')
    objeto.select_set(True)
    bpy.context.view_layer.objects.active = objeto

    return objeto


if __name__ == "__main__":
    generar_preroll_lowpoly()
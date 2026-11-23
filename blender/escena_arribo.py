"""Escena 1 · "Arribo" — sargazo derivando y acumulándose en una playa del Caribe mexicano.

Autor: Gersain Barrón · Proyecto Sargazo IA
Ejecutar dentro de Blender 5.x:
    exec(open(r"<ruta>/sargazo-ia/blender/escena_arribo.py", encoding="utf8").read())

Elementos:
  - Océano con modificador Ocean (oleaje, espuma) y agua turquesa (transmisión + absorción volumétrica).
  - Fondo/playa de arena con pendiente: lo somero se ve turquesa, lo profundo azul.
  - Balsas de sargazo (Geometry Nodes): puntos estables que derivan hacia la costa, flotan sobre las olas
    (raycast contra el océano) y "crecen" con el tiempo (el umbral de la máscara baja → manchas más grandes).
  - Acumulación en la playa: franja de sargazo seco que aumenta con el tiempo.
  - Cielo físico al atardecer y cámara a ras de playa.
"""
import math
import random

import bmesh
import bpy
from mathutils import Matrix, Vector

FPS, FRAMES = 24, 240
COSTA_Y = 6.0          # línea de costa (z = 0)
PENDIENTE = 0.06       # pendiente de la playa/fondo
VEL_DERIVA = 1.2       # m/s de deriva (time-lapse)

random.seed(7)
esc = bpy.context.scene

# ---------------------------------------------------------------- limpiar
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
for c in list(bpy.data.collections):
    bpy.data.collections.remove(c)
for ng in list(bpy.data.node_groups):
    bpy.data.node_groups.remove(ng)
for m in list(bpy.data.materials):
    bpy.data.materials.remove(m)
for me in list(bpy.data.meshes):
    bpy.data.meshes.remove(me)


def col(nombre, padre=None):
    c = bpy.data.collections.new(nombre)
    (padre or esc.collection).children.link(c)
    return c


def ins(n, nombre, i=0):
    s = [s for s in n.inputs if s.name == nombre and s.enabled]
    return s[i]


ALIAS = {"Fac": ("Fac", "Factor", "Value"), "Factor": ("Factor", "Fac")}


def outs(n, nombre, i=0):
    for alias in ALIAS.get(nombre, (nombre,)):
        s = [s for s in n.outputs if s.name == alias and s.enabled]
        if s:
            return s[i]
    raise KeyError(f"{n.bl_idname} no tiene salida {nombre}: {[o.name for o in n.outputs if o.enabled]}")


# ---------------------------------------------------------------- materiales
def principled(nt):
    """Nodo Principled del material (los nombres de nodo pueden venir traducidos)."""
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    salida = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
    nt.links.new(p.outputs[0], salida.inputs["Surface"])
    return p


def mat_agua():
    m = bpy.data.materials.new("Agua_Caribe"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (0.85, 0.97, 1.0, 1)
    p.inputs["Roughness"].default_value = 0.025
    p.inputs["IOR"].default_value = 1.333
    p.inputs["Transmission Weight"].default_value = 1.0
    espuma = nt.nodes.new("ShaderNodeBsdfPrincipled")
    espuma.inputs["Base Color"].default_value = (0.95, 0.97, 0.98, 1)
    espuma.inputs["Roughness"].default_value = 0.6
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "foam"
    rampa = nt.nodes.new("ShaderNodeValToRGB")
    rampa.color_ramp.elements[0].position = 0.45; rampa.color_ramp.elements[1].position = 0.9
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(outs(at, "Fac"), rampa.inputs[0])
    nt.links.new(rampa.outputs["Color"], mix.inputs[0])
    nt.links.new(p.outputs[0], mix.inputs[1]); nt.links.new(espuma.outputs[0], mix.inputs[2])
    vol = nt.nodes.new("ShaderNodeVolumeAbsorption")
    vol.inputs["Color"].default_value = (0.18, 0.78, 0.74, 1)
    vol.inputs["Density"].default_value = 0.35
    nt.links.new(mix.outputs[0], o.inputs["Surface"]); nt.links.new(vol.outputs[0], o.inputs["Volume"])
    return m


def mat_arena():
    m = bpy.data.materials.new("Arena"); m.use_nodes = True
    nt = m.node_tree; p = principled(nt)
    ruido = nt.nodes.new("ShaderNodeTexNoise"); ruido.inputs["Scale"].default_value = 40
    rampa = nt.nodes.new("ShaderNodeValToRGB")
    rampa.color_ramp.elements[0].color = (0.55, 0.47, 0.34, 1)
    rampa.color_ramp.elements[1].color = (0.70, 0.62, 0.47, 1)
    nt.links.new(outs(ruido, "Fac"), rampa.inputs[0])
    nt.links.new(rampa.outputs["Color"], p.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.25
    r2 = nt.nodes.new("ShaderNodeTexNoise"); r2.inputs["Scale"].default_value = 300
    nt.links.new(outs(r2, "Fac"), bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    p.inputs["Roughness"].default_value = 0.9
    return m


def mat_sargazo(nombre, c1, c2, sss=0.25):
    m = bpy.data.materials.new(nombre); m.use_nodes = True
    nt = m.node_tree; p = principled(nt)
    ruido = nt.nodes.new("ShaderNodeTexNoise"); ruido.inputs["Scale"].default_value = 25
    rampa = nt.nodes.new("ShaderNodeValToRGB")
    rampa.color_ramp.elements[0].color = c1; rampa.color_ramp.elements[1].color = c2
    nt.links.new(outs(ruido, "Fac"), rampa.inputs[0])
    nt.links.new(rampa.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.45
    p.inputs["Subsurface Weight"].default_value = sss
    p.inputs["Subsurface Radius"].default_value = (0.02, 0.015, 0.005)
    return m


M_AGUA, M_ARENA = mat_agua(), mat_arena()
M_SARG = mat_sargazo("Sargazo_Vivo", (0.20, 0.10, 0.012, 1), (0.42, 0.24, 0.03, 1))
M_SECO = mat_sargazo("Sargazo_Seco", (0.05, 0.03, 0.012, 1), (0.13, 0.08, 0.03, 1), sss=0.03)

# ---------------------------------------------------------------- variantes de "mata" de sargazo
C_VAR = col("Sargazo_Variantes")
C_VAR.hide_render = False
for v in range(4):
    bm = bmesh.new()
    for _ in range(45):  # vesículas (flotadores)
        r = random.uniform(0.012, 0.03)
        pos = Vector((random.gauss(0, 0.11), random.gauss(0, 0.08), random.uniform(-0.01, 0.025)))
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r, matrix=Matrix.Translation(pos))
    for _ in range(16):  # hojas alargadas
        pos = Vector((random.gauss(0, 0.12), random.gauss(0, 0.09), random.uniform(-0.01, 0.02)))
        rot = Matrix.Rotation(random.uniform(0, math.tau), 4, "Z") @ Matrix.Rotation(random.uniform(-.4, .4), 4, "X")
        esc_m = Matrix.Diagonal((random.uniform(0.05, 0.09), 0.012, 0.004, 1))
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1, matrix=Matrix.Translation(pos) @ rot @ esc_m)
    me = bpy.data.meshes.new(f"sargazo_{v}"); bm.to_mesh(me); bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    me.materials.append(M_SARG)
    ob = bpy.data.objects.new(f"Sargazo_{v}", me)
    C_VAR.objects.link(ob)
# La colección de variantes no se ve directamente (solo como instancias)
lc = bpy.context.view_layer.layer_collection.children[C_VAR.name]
lc.exclude = True

# ---------------------------------------------------------------- terreno (fondo + playa)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=240, y_subdivisions=180, size=1)
terreno = bpy.context.active_object; terreno.name = "Playa_Fondo"
terreno.scale = (140, 110, 1); bpy.ops.object.transform_apply(scale=True)
terreno.location.y = -15
bpy.ops.object.transform_apply(location=True)
for vtx in terreno.data.vertices:
    y = vtx.co.y
    z = (y - COSTA_Y) * PENDIENTE
    if y > COSTA_Y + 10:        # berma de playa: más plana
        z = 10 * PENDIENTE + (y - COSTA_Y - 10) * 0.02
    vtx.co.z = max(z, -3.5) + 0.05 * math.sin(vtx.co.x * 0.7) * math.sin(y * 0.5)
terreno.data.materials.append(M_ARENA)
bpy.ops.object.shade_smooth()

# ---------------------------------------------------------------- océano
# El modificador genera la malla desde una esquina: se desplaza para cubrir x −70…70, y −115…25
bpy.ops.mesh.primitive_plane_add(size=1, location=(-35, -80, 0))
oceano = bpy.context.active_object; oceano.name = "Oceano"
mo = oceano.modifiers.new("Ocean", "OCEAN")
mo.geometry_mode = "GENERATE"
mo.spatial_size = 70; mo.repeat_x = 2; mo.repeat_y = 2
mo.resolution = 16
mo.wave_scale = 0.35; mo.choppiness = 0.9; mo.wind_velocity = 7; mo.wave_alignment = 0.4
mo.wave_direction = math.radians(90)
mo.use_normals = True
mo.use_foam = True; mo.foam_layer_name = "foam"; mo.foam_coverage = 0.15
fc = mo.driver_add("time"); fc.driver.expression = f"frame/{FPS}"
oceano.data.materials.append(M_AGUA)
bpy.ops.object.shade_smooth()


# ---------------------------------------------------------------- Geometry Nodes: balsas a la deriva
def nodo(nt, tipo, x, y, **props):
    n = nt.nodes.new(tipo); n.location = (x, y)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def gn_balsas():
    ng = bpy.data.node_groups.new("GN_Balsas_Sargazo", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nt, L = ng, ng.links.new
    gi = nodo(nt, "NodeGroupInput", -1600, 0); go = nodo(nt, "NodeGroupOutput", 1400, 0)

    dist = nodo(nt, "GeometryNodeDistributePointsOnFaces", -1400, 0, distribute_method="RANDOM")
    ins(dist, "Density").default_value = 30; ins(dist, "Seed").default_value = 3
    L(gi.outputs[0], ins(dist, "Mesh"))

    tiempo = nodo(nt, "GeometryNodeInputSceneTime", -1400, -400)
    pos = nodo(nt, "GeometryNodeInputPosition", -1400, -250)

    # Máscara de manchas (estable: evaluada en la posición inicial de cada punto)
    # Franjas alargadas paralelas a la costa (como las hileras que forma el viento)
    estirar = nodo(nt, "ShaderNodeVectorMath", -1300, -250, operation="MULTIPLY")
    L(pos.outputs[0], estirar.inputs[0]); estirar.inputs[1].default_value = (0.045, 0.17, 0.0)
    ruido = nodo(nt, "ShaderNodeTexNoise", -1150, -250)
    ins(ruido, "Scale").default_value = 1.0; ins(ruido, "Detail").default_value = 4
    ins(ruido, "Roughness").default_value = 0.62
    L(estirar.outputs[0], ins(ruido, "Vector"))
    # Umbral que baja con el tiempo → manchas que crecen
    umbral = nodo(nt, "ShaderNodeMapRange", -1150, -500, clamp=True)
    L(outs(tiempo, "Seconds"), ins(umbral, "Value"))
    ins(umbral, "From Min").default_value = 0; ins(umbral, "From Max").default_value = FRAMES / FPS
    ins(umbral, "To Min").default_value = 0.60; ins(umbral, "To Max").default_value = 0.47
    umbral_mas = nodo(nt, "ShaderNodeMath", -950, -560, operation="ADD")
    L(umbral.outputs[0], umbral_mas.inputs[0]); umbral_mas.inputs[1].default_value = 0.09
    dens = nodo(nt, "ShaderNodeMapRange", -800, -300, clamp=True)
    L(outs(ruido, "Fac"), ins(dens, "Value"))
    L(umbral.outputs[0], ins(dens, "From Min")); L(umbral_mas.outputs[0], ins(dens, "From Max"))
    ins(dens, "To Min").default_value = 0; ins(dens, "To Max").default_value = 1
    guardar_dens = nodo(nt, "GeometryNodeStoreNamedAttribute", -1150, 0, data_type="FLOAT", domain="POINT")
    ins(guardar_dens, "Name").default_value = "dens"
    L(outs(dist, "Points"), ins(guardar_dens, "Geometry")); L(dens.outputs[0], ins(guardar_dens, "Value"))

    # Conservar el punto si su valor aleatorio < densidad (aparecen de forma estable al crecer)
    azar = nodo(nt, "FunctionNodeRandomValue", -800, -100, data_type="FLOAT")
    ins(azar, "Seed").default_value = 11
    menor = nodo(nt, "FunctionNodeCompare", -600, -150, data_type="FLOAT", operation="GREATER_EQUAL")
    L(outs(azar, "Value"), ins(menor, "A")); L(dens.outputs[0], ins(menor, "B"))
    borrar = nodo(nt, "GeometryNodeDeleteGeometry", -400, 0, domain="POINT")
    L(guardar_dens.outputs[0], ins(borrar, "Geometry")); L(outs(menor, "Result"), ins(borrar, "Selection"))

    # Deriva hacia la costa (+Y)
    desp_y = nodo(nt, "ShaderNodeMath", -600, -400, operation="MULTIPLY")
    L(outs(tiempo, "Seconds"), desp_y.inputs[0]); desp_y.inputs[1].default_value = VEL_DERIVA
    vec = nodo(nt, "ShaderNodeCombineXYZ", -400, -400); L(desp_y.outputs[0], ins(vec, "Y"))
    mover = nodo(nt, "GeometryNodeSetPosition", -200, 0)
    L(borrar.outputs[0], ins(mover, "Geometry")); L(vec.outputs[0], ins(mover, "Offset"))

    # Eliminar lo que ya llegó a la costa
    pos2 = nodo(nt, "GeometryNodeInputPosition", -200, -250)
    sep = nodo(nt, "ShaderNodeSeparateXYZ", 0, -250); L(pos2.outputs[0], sep.inputs[0])
    en_costa = nodo(nt, "FunctionNodeCompare", 200, -250, data_type="FLOAT", operation="GREATER_THAN")
    L(outs(sep, "Y"), ins(en_costa, "A")); ins(en_costa, "B").default_value = COSTA_Y - 0.5
    borrar2 = nodo(nt, "GeometryNodeDeleteGeometry", 200, 0, domain="POINT")
    L(mover.outputs[0], ins(borrar2, "Geometry")); L(outs(en_costa, "Result"), ins(borrar2, "Selection"))

    # Flotar: raycast vertical contra el océano deformado
    info = nodo(nt, "GeometryNodeObjectInfo", 200, 300, transform_space="RELATIVE")
    ins(info, "Object").default_value = oceano
    alto = nodo(nt, "ShaderNodeCombineXYZ", 200, -450); ins(alto, "Z").default_value = 20
    origen = nodo(nt, "ShaderNodeVectorMath", 400, -400, operation="ADD")
    pos3 = nodo(nt, "GeometryNodeInputPosition", 200, -550)
    L(pos3.outputs[0], origen.inputs[0]); L(alto.outputs[0], origen.inputs[1])
    rayo = nodo(nt, "GeometryNodeRaycast", 600, 300)
    L(outs(info, "Geometry"), ins(rayo, "Target Geometry"))
    L(origen.outputs[0], ins(rayo, "Source Position"))
    ins(rayo, "Ray Direction").default_value = (0, 0, -1); ins(rayo, "Ray Length").default_value = 60
    flotar = nodo(nt, "GeometryNodeSetPosition", 800, 0)
    L(borrar2.outputs[0], ins(flotar, "Geometry")); L(outs(rayo, "Hit Position"), ins(flotar, "Position"))
    L(outs(rayo, "Is Hit"), ins(flotar, "Selection"))

    # Instancias: variante al azar, giro al azar, tamaño según densidad (crecimiento)
    coleccion = nodo(nt, "GeometryNodeCollectionInfo", 800, 400)
    ins(coleccion, "Collection").default_value = C_VAR
    ins(coleccion, "Separate Children").default_value = True; ins(coleccion, "Reset Children").default_value = True
    idx = nodo(nt, "FunctionNodeRandomValue", 800, -300, data_type="INT")
    ins(idx, "Min").default_value = 0; ins(idx, "Max").default_value = 3; ins(idx, "Seed").default_value = 5
    giro = nodo(nt, "FunctionNodeRandomValue", 800, -500, data_type="FLOAT")
    ins(giro, "Max").default_value = math.tau; ins(giro, "Seed").default_value = 9
    euler = nodo(nt, "ShaderNodeCombineXYZ", 1000, -500); L(outs(giro, "Value"), ins(euler, "Z"))
    rot = nodo(nt, "FunctionNodeEulerToRotation", 1100, -500); L(euler.outputs[0], rot.inputs[0])
    leer_dens = nodo(nt, "GeometryNodeInputNamedAttribute", 800, -700, data_type="FLOAT")
    ins(leer_dens, "Name").default_value = "dens"
    escala = nodo(nt, "ShaderNodeMapRange", 1000, -700)
    L(outs(leer_dens, "Attribute"), ins(escala, "Value"))
    ins(escala, "To Min").default_value = 0.7; ins(escala, "To Max").default_value = 1.6
    inst = nodo(nt, "GeometryNodeInstanceOnPoints", 1200, 0)
    L(flotar.outputs[0], ins(inst, "Points")); L(outs(coleccion, "Instances"), ins(inst, "Instance"))
    ins(inst, "Pick Instance").default_value = True
    L(outs(idx, "Value"), ins(inst, "Instance Index")); L(rot.outputs[0], ins(inst, "Rotation"))
    L(escala.outputs[0], ins(inst, "Scale"))
    L(inst.outputs[0], go.inputs[0])
    return ng


bpy.ops.mesh.primitive_plane_add(size=1, location=(0, -30, 0))
balsas = bpy.context.active_object; balsas.name = "Sargazo_Balsas"
balsas.scale = (90, 70, 1); bpy.ops.object.transform_apply(scale=True, location=False)
# El plano emisor queda "detrás" (y de −65 a +5) y los puntos derivan hacia la costa
balsas.location = (0, 0, 0)
for vtx in balsas.data.vertices:
    vtx.co.y += -30
balsas.modifiers.new("Balsas", "NODES").node_group = gn_balsas()


# ---------------------------------------------------------------- Geometry Nodes: acumulación en playa
def gn_playa():
    ng = bpy.data.node_groups.new("GN_Sargazo_Playa", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nt, L = ng, ng.links.new
    gi = nodo(nt, "NodeGroupInput", -1200, 0); go = nodo(nt, "NodeGroupOutput", 1200, 0)
    dist = nodo(nt, "GeometryNodeDistributePointsOnFaces", -1000, 0, distribute_method="RANDOM")
    ins(dist, "Density").default_value = 160; ins(dist, "Seed").default_value = 21
    L(gi.outputs[0], ins(dist, "Mesh"))
    tiempo = nodo(nt, "GeometryNodeInputSceneTime", -1000, -400)
    pos = nodo(nt, "GeometryNodeInputPosition", -1000, -250)
    sep = nodo(nt, "ShaderNodeSeparateXYZ", -800, -250); L(pos.outputs[0], sep.inputs[0])
    # Más sargazo pegado a la orilla, y más con el tiempo
    d_orilla = nodo(nt, "ShaderNodeMapRange", -600, -250, clamp=True)
    L(outs(sep, "Y"), ins(d_orilla, "Value"))
    ins(d_orilla, "From Min").default_value = COSTA_Y; ins(d_orilla, "From Max").default_value = COSTA_Y + 2.5
    ins(d_orilla, "To Min").default_value = 1; ins(d_orilla, "To Max").default_value = 0
    crec = nodo(nt, "ShaderNodeMapRange", -600, -450, clamp=True)
    L(outs(tiempo, "Seconds"), ins(crec, "Value"))
    ins(crec, "From Min").default_value = 1.5; ins(crec, "From Max").default_value = FRAMES / FPS
    ruido = nodo(nt, "ShaderNodeTexNoise", -800, -600); ins(ruido, "Scale").default_value = 0.35
    L(pos.outputs[0], ins(ruido, "Vector"))
    m1 = nodo(nt, "ShaderNodeMath", -400, -300, operation="MULTIPLY")
    L(d_orilla.outputs[0], m1.inputs[0]); L(crec.outputs[0], m1.inputs[1])
    m2 = nodo(nt, "ShaderNodeMath", -250, -400, operation="MULTIPLY")
    L(m1.outputs[0], m2.inputs[0]); L(outs(ruido, "Fac"), m2.inputs[1])
    m3 = nodo(nt, "ShaderNodeMath", -100, -400, operation="MULTIPLY"); L(m2.outputs[0], m3.inputs[0])
    m3.inputs[1].default_value = 1.8
    azar = nodo(nt, "FunctionNodeRandomValue", -250, -100, data_type="FLOAT"); ins(azar, "Seed").default_value = 4
    cmp = nodo(nt, "FunctionNodeCompare", 0, -150, data_type="FLOAT", operation="GREATER_EQUAL")
    L(outs(azar, "Value"), ins(cmp, "A")); L(m3.outputs[0], ins(cmp, "B"))
    borrar = nodo(nt, "GeometryNodeDeleteGeometry", 150, 0, domain="POINT")
    L(outs(dist, "Points"), ins(borrar, "Geometry")); L(outs(cmp, "Result"), ins(borrar, "Selection"))
    # Apoyar sobre la arena
    info = nodo(nt, "GeometryNodeObjectInfo", 150, 300, transform_space="RELATIVE")
    ins(info, "Object").default_value = terreno
    pos2 = nodo(nt, "GeometryNodeInputPosition", 150, -500)
    alto = nodo(nt, "ShaderNodeVectorMath", 300, -500, operation="ADD")
    L(pos2.outputs[0], alto.inputs[0]); alto.inputs[1].default_value = (0, 0, 10)
    rayo = nodo(nt, "GeometryNodeRaycast", 450, 300)
    L(outs(info, "Geometry"), ins(rayo, "Target Geometry")); L(alto.outputs[0], ins(rayo, "Source Position"))
    ins(rayo, "Ray Direction").default_value = (0, 0, -1); ins(rayo, "Ray Length").default_value = 30
    apoyar = nodo(nt, "GeometryNodeSetPosition", 650, 0)
    L(borrar.outputs[0], ins(apoyar, "Geometry")); L(outs(rayo, "Hit Position"), ins(apoyar, "Position"))
    coleccion = nodo(nt, "GeometryNodeCollectionInfo", 650, 400)
    ins(coleccion, "Collection").default_value = C_VAR
    ins(coleccion, "Separate Children").default_value = True; ins(coleccion, "Reset Children").default_value = True
    idx = nodo(nt, "FunctionNodeRandomValue", 650, -300, data_type="INT")
    ins(idx, "Max").default_value = 3; ins(idx, "Seed").default_value = 8
    giro = nodo(nt, "FunctionNodeRandomValue", 650, -500, data_type="FLOAT"); ins(giro, "Max").default_value = math.tau
    euler = nodo(nt, "ShaderNodeCombineXYZ", 800, -500); L(outs(giro, "Value"), ins(euler, "Z"))
    rot = nodo(nt, "FunctionNodeEulerToRotation", 900, -500); L(euler.outputs[0], rot.inputs[0])
    inst = nodo(nt, "GeometryNodeInstanceOnPoints", 900, 0)
    L(apoyar.outputs[0], ins(inst, "Points")); L(outs(coleccion, "Instances"), ins(inst, "Instance"))
    ins(inst, "Pick Instance").default_value = True
    L(outs(idx, "Value"), ins(inst, "Instance Index")); L(rot.outputs[0], ins(inst, "Rotation"))
    ins(inst, "Scale").default_value = (1.3, 1.3, 0.55)
    mat = nodo(nt, "GeometryNodeSetMaterial", 1050, 0)
    L(inst.outputs[0], ins(mat, "Geometry")); ins(mat, "Material").default_value = M_SECO
    L(mat.outputs[0], go.inputs[0])
    return ng


bpy.ops.mesh.primitive_plane_add(size=1, location=(0, COSTA_Y + 2.0, 0.3))
franja = bpy.context.active_object; franja.name = "Sargazo_Playa"
franja.scale = (70, 4.5, 1); bpy.ops.object.transform_apply(scale=True)
franja.modifiers.new("Playa", "NODES").node_group = gn_playa()

# ---------------------------------------------------------------- cielo, cámara, render
mundo = bpy.data.worlds.get("World") or bpy.data.worlds.new("World"); esc.world = mundo
mundo.use_nodes = True; wn = mundo.node_tree; wn.nodes.clear()
cielo = wn.nodes.new("ShaderNodeTexSky")
for tipo in ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING"):
    try:
        cielo.sky_type = tipo; break
    except Exception:
        pass
cielo.sun_elevation = math.radians(28); cielo.sun_rotation = math.radians(30)  # sol detrás/al lado de la cámara
fondo = wn.nodes.new("ShaderNodeBackground"); fondo.inputs["Strength"].default_value = 0.25
salida = wn.nodes.new("ShaderNodeOutputWorld")
wn.links.new(cielo.outputs[0], fondo.inputs[0]); wn.links.new(fondo.outputs[0], salida.inputs[0])

cam_data = bpy.data.cameras.new("Camara"); cam_data.lens = 26
cam = bpy.data.objects.new("Camara", cam_data); esc.collection.objects.link(cam)
cam.location = (5.0, COSTA_Y + 9.0, 4.2)
direccion = Vector((-2.0, -16.0, -1.2)) - cam.location
cam.rotation_euler = direccion.to_track_quat("-Z", "Y").to_euler()
esc.camera = cam

esc.render.engine = "CYCLES"
esc.cycles.device = "GPU"
esc.cycles.samples = 96
esc.cycles.use_denoising = True
esc.cycles.max_bounces = 12; esc.cycles.transmission_bounces = 12; esc.cycles.volume_bounces = 2
esc.render.resolution_x, esc.render.resolution_y = 1920, 1080
esc.render.fps = FPS; esc.frame_start, esc.frame_end = 1, FRAMES
esc.view_settings.view_transform = "AgX"
esc.view_settings.exposure = -0.5
esc.view_settings.look = "AgX - Medium High Contrast" if "AgX - Medium High Contrast" in [
    i.identifier for i in esc.view_settings.bl_rna.properties["look"].enum_items] else "None"
esc.frame_set(1)
print("Escena 'Arribo' lista")

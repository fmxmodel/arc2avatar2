import bpy
import os

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import PLY
ply_path = os.path.join(os.path.dirname(__file__), 'subject_standard.ply')
bpy.ops.wm.ply_import(filepath=ply_path)

# Get the imported object
obj = bpy.context.active_object

# Switch to Material Preview view, set clip end far
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        space = area.spaces[0]
        space.shading.type = 'MATERIAL'
        space.clip_end = 1000
        # Rotate view for a nice initial angle
        space.region_3d.view_rotation = (0.5, -0.3, 0.3, 0.7)

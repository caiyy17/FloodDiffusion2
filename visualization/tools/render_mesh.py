"""Pyrender-based SMPL mesh rendering with orthographic camera.

Renders mesh frames that visually align with the skeleton renderer's
camera projection, so skeleton can be overlaid on top.
"""

import math
import os

import numpy as np

# numpy 2.x compat fix for pyrender 0.1.45
if not hasattr(np, "infty"):
    np.infty = np.inf

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from .egl_setup import configure_egl_vendor

configure_egl_vendor()

import pyrender
import trimesh


def render_mesh_frames(verts, faces, cam_params):
    """Render SMPL mesh to a list of RGB image frames.

    Args:
        verts: (T, V, 3) mesh vertices in world coordinates.
        faces: (F, 3) triangle face indices.
        cam_params: dict from compute_camera_params() containing camera
                    geometry that matches the skeleton renderer.

    Returns:
        list of np.ndarray images (H, W, 3), uint8.
    """
    T = verts.shape[0]
    width = cam_params["width"]
    height = cam_params["height"]
    look_at = cam_params["look_at"]
    distance = cam_params["distance"]
    elevation = cam_params["elevation"]
    azimuth = cam_params["azimuth"]
    ortho_scale = cam_params["ortho_scale"]

    # Orthographic camera with magnification matching skeleton projection
    # The skeleton renderer uses screen_scale = min(W,H) * 0.4 / ortho_scale
    # For pyrender ortho, xmag=ymag controls the visible half-width in world units
    # To match: xmag = ortho_scale * 1.25 (empirically calibrated)
    xmag = ymag = ortho_scale * 1.25

    # Compute camera pose (same math as skeleton renderer)
    front = np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.sin(elevation),
        math.cos(elevation) * math.sin(azimuth),
    ])
    front /= np.linalg.norm(front)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(front, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, front)

    # Pyrender camera looks along local -Z. Place camera so that -Z points
    # toward the scene: cam_pos = look_at - front * distance, cam_z = -front.
    # (In orthographic projection, shifting cam_pos along the look direction
    # doesn't change the image, so this is compatible with the skeleton
    # renderer's cam_pos = look_at + front * distance.)
    cam_pos = look_at - front * distance
    cam_z = -front

    R = np.stack([right, up, cam_z], axis=1)  # (3, 3)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R
    cam_pose[:3, 3] = cam_pos

    # Body material: beige skin tone
    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.55, 0.45, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )

    # Ground plane material
    ground_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )

    # Ground plane covering motion extent
    x_min = cam_params["x_min"]
    x_max = cam_params["x_max"]
    z_min = cam_params["z_min"]
    z_max = cam_params["z_max"]
    padding = 1.0
    gv = np.array([
        [x_min - padding, 0, z_min - padding],
        [x_max + padding, 0, z_min - padding],
        [x_max + padding, 0, z_max + padding],
        [x_min - padding, 0, z_max + padding],
    ], dtype=np.float32)
    gf = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    ground_tri = trimesh.Trimesh(vertices=gv, faces=gf, process=False)
    ground_mesh = pyrender.Mesh.from_trimesh(ground_tri, material=ground_material, smooth=True)

    # Build scene
    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],
        ambient_light=[0.4, 0.4, 0.4],
    )
    scene.add(ground_mesh)

    # Lights
    main_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    main_light_pose = np.array([
        [1, 0, 0, 0],
        [0, 0.8, -0.6, 0],
        [0, 0.6, 0.8, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    scene.add(main_light, pose=main_light_pose)

    fill_light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=2.0)
    fill_light_pose = np.array([
        [1, 0, 0, 0],
        [0, 0.6, 0.8, 0],
        [0, -0.8, 0.6, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    scene.add(fill_light, pose=fill_light_pose)

    # Camera
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    scene.add(camera, pose=cam_pose, name="ortho_cam")

    # Initial body mesh
    body_tri = trimesh.Trimesh(vertices=verts[0], faces=faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_tri, material=body_material, smooth=True)
    body_node = scene.add(body_mesh)

    # Renderer
    renderer = pyrender.OffscreenRenderer(width, height)
    render_flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL

    images = []
    try:
        for t in range(T):
            # Update body mesh
            scene.remove_node(body_node)
            body_tri = trimesh.Trimesh(vertices=verts[t], faces=faces, process=False)
            body_mesh = pyrender.Mesh.from_trimesh(body_tri, material=body_material, smooth=True)
            body_node = scene.add(body_mesh)

            color, _ = renderer.render(scene, flags=render_flags)
            images.append(color)
    finally:
        renderer.delete()

    return images

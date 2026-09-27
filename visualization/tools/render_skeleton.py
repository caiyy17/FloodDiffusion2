#!/usr/bin/env python3
import math

import numpy as np


def get_smpl22_chains():
    return [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]


def get_chain_color_table():
    """Normalized RGB palette used to color consecutive bones."""
    return [
        [254 / 255, 178 / 255, 26 / 255],  # orange
        [0 / 255, 170 / 255, 255 / 255],  # cyan
        [19 / 255, 70 / 255, 134 / 255],  # aquamarine
        [255 / 255, 182 / 255, 0 / 255],  # amber
        [0 / 255, 212 / 255, 126 / 255],  # aquamarine
    ]


def compute_camera_params(data):
    """Compute camera parameters from joint position data.

    These parameters fully describe the orthographic camera used by the
    skeleton renderer, so that a mesh renderer can produce pixel-aligned
    images for overlay compositing.

    Args:
        data: (T, J, 3) joint positions.

    Returns:
        dict with keys: look_at, distance, elevation, azimuth,
        sk_height, motion_scale, ortho_scale, screen_scale,
        width, height, x_min, x_max, y_min, y_max, z_min, z_max.
    """
    all_points = data.reshape(-1, 3)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    x_range = x_max - x_min
    z_range = z_max - z_min
    y_range = y_max - y_min
    horizontal_range = max(x_range, z_range)

    width, height = 480, 480

    elevation = -math.pi / 10.0
    azimuth = -math.pi * 3.0 / 4.0

    sk_height = y_range if y_range > 1.0 else 1.5

    motion_ratio = horizontal_range / sk_height
    if motion_ratio > 1.5:
        motion_scale = 1.0 + (motion_ratio - 1.5) * 0.5
    else:
        motion_scale = 1.0

    distance = sk_height * 3.0
    look_at = np.array(
        [(x_min + x_max) / 2, y_min + sk_height * 0.45, (z_min + z_max) / 2]
    )

    ortho_scale = sk_height * 0.8 * motion_scale
    screen_scale = min(width, height) * 0.4 / ortho_scale

    return {
        "look_at": look_at,
        "distance": distance,
        "elevation": elevation,
        "azimuth": azimuth,
        "sk_height": sk_height,
        "motion_scale": motion_scale,
        "ortho_scale": ortho_scale,
        "screen_scale": screen_scale,
        "width": width,
        "height": height,
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
    }


def render_skeleton_frames(data, chains, canvas_images=None):
    """Render skeleton joint data to a list of image frames.

    Args:
        data: (T, J, 3) joint positions.
        chains: list of joint chains for bone drawing.
        canvas_images: optional list of np.ndarray (H, W, 3) uint8 images
            to draw skeleton on top of. When None, uses white background.

    Returns:
        list of np.ndarray images (H, W, 3), uint8.
    """
    traj = data[:, 0, [0, 2]]  # root joint XZ trajectory

    cam = compute_camera_params(data)
    width = cam["width"]
    height = cam["height"]

    center_x = width // 2
    center_z = height // 2

    bone_colors = get_chain_color_table()

    def to_uint8_palette(colors):
        converted = []
        for color in colors:
            arr = np.array(color, dtype=np.float32)
            if arr.size < 3:
                arr = np.pad(
                    arr, (0, 3 - arr.size), mode="constant", constant_values=0.0
                )
            arr = np.clip(arr[:3], 0.0, 1.0)
            converted.append((arr * 255).astype(np.uint8).tolist())
        return converted

    bone_colors_uint8 = to_uint8_palette(bone_colors)

    # Compute camera vectors once
    front = np.array(
        [
            math.cos(cam["elevation"]) * math.cos(cam["azimuth"]),
            math.sin(cam["elevation"]),
            math.cos(cam["elevation"]) * math.sin(cam["azimuth"]),
        ]
    )
    front /= np.linalg.norm(front)
    cam_pos = cam["look_at"] + front * cam["distance"]
    up = np.array([0, 1, 0])
    right = np.cross(front, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, front)

    screen_scale = cam["screen_scale"]

    def world_to_screen(point):
        to_point = np.array(point) - cam_pos
        x_cam = np.dot(to_point, right)
        y_cam = np.dot(to_point, up)
        screen_x = int(center_x + x_cam * screen_scale)
        screen_y = int(center_z - y_cam * screen_scale)
        return (screen_x, screen_y)

    def draw_line_vectorized(img, p1, p2, color, thickness=2):
        x1, y1 = p1
        x2, y2 = p2
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy)

        if steps == 0:
            return

        # Vectorized line generation
        t = np.linspace(0, 1, steps + 1)
        x_coords = (x1 + t * (x2 - x1)).astype(np.int32)
        y_coords = (y1 + t * (y2 - y1)).astype(np.int32)

        # Create thickness offsets
        half_thick = thickness // 2
        offsets = np.arange(-half_thick, half_thick + 1)
        dx_offsets, dy_offsets = np.meshgrid(offsets, offsets, indexing="ij")
        dx_offsets = dx_offsets.flatten()
        dy_offsets = dy_offsets.flatten()

        # Broadcast coordinates with thickness offsets
        x_thick = x_coords[:, None] + dx_offsets[None, :]
        y_thick = y_coords[:, None] + dy_offsets[None, :]

        # Flatten and filter valid coordinates
        x_flat = x_thick.flatten()
        y_flat = y_thick.flatten()

        # Bounds checking
        valid_mask = (
            (x_flat >= 0) & (x_flat < width) & (y_flat >= 0) & (y_flat < height)
        )
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]

        # Vectorized assignment
        img[y_valid, x_valid] = color

    def draw_circle_vectorized(img, center, radius, color):
        cx, cy = center
        cx = max(0, min(width - 1, cx))
        cy = max(0, min(height - 1, cy))

        # Create coordinate grids for the bounding box
        y_lo = max(0, cy - radius)
        y_hi = min(height, cy + radius + 1)
        x_lo = max(0, cx - radius)
        x_hi = min(width, cx + radius + 1)

        if y_lo >= y_hi or x_lo >= x_hi:
            return

        # Vectorized distance calculation
        y_coords, x_coords = np.meshgrid(
            np.arange(y_lo, y_hi), np.arange(x_lo, x_hi), indexing="ij"
        )

        # Calculate squared distances from center
        dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2

        # Create mask for pixels inside circle
        circle_mask = dist_sq <= radius**2

        # Apply color to pixels inside circle
        img[y_coords[circle_mask], x_coords[circle_mask]] = color

    images = []
    for frame in range(len(data)):
        if canvas_images is not None:
            img = canvas_images[frame].copy()
        else:
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
        joints = data[frame]
        if frame > 0:
            for i in range(frame):
                if i + 1 < len(traj):
                    p1 = world_to_screen([traj[i, 0], 0, traj[i, 1]])
                    p2 = world_to_screen([traj[i + 1, 0], 0, traj[i + 1, 1]])
                    draw_line_vectorized(
                        img, p1, p2, [255, 0, 0], thickness=3
                    )  # Red trajectory
        # Draw bones with palette cycling per segment
        color_index = 0
        for chain in chains:
            for i in range(len(chain) - 1):
                if chain[i] < len(joints) and chain[i + 1] < len(joints):
                    p1 = world_to_screen(joints[chain[i]])
                    p2 = world_to_screen(joints[chain[i + 1]])
                    draw_line_vectorized(
                        img,
                        p1,
                        p2,
                        bone_colors_uint8[color_index % len(bone_colors_uint8)],
                        thickness=4,
                    )
            color_index += 1
        # Draw joints (blue circles)
        for joint in joints:
            center = world_to_screen(joint)
            draw_circle_vectorized(img, center, 3, [0, 100, 255])  # Blue joints
        images.append(img)

    return images


def main():
    data = np.random.rand(60, 22, 3)
    frames = render_skeleton_frames(data, get_smpl22_chains())
    print(f"Rendered {len(frames)} frames, shape: {frames[0].shape}")


if __name__ == "__main__":
    main()

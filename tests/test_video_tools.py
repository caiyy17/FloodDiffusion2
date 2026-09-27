"""Exercise rendering's bundled media tools with a real encoded video."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import imageio_ffmpeg
import numpy as np

from visualization import visualize


class VideoToolsTests(unittest.TestCase):
    def test_metadata_and_composite_without_system_media_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "composite"
            source.mkdir()
            clip = source / "motion.mp4"
            width, height, fps, frame_count = 320, 160, 20, 12
            writer = imageio_ffmpeg.write_frames(
                str(clip), (width, height), fps=fps, codec="libx264"
            )
            writer.send(None)
            try:
                for index in range(frame_count):
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    frame[:, :, 0] = 50 + 10 * index
                    writer.send(frame)
            finally:
                writer.close()

            # Disable both discovery locations, while leaving real encoding,
            # metadata reads, and composition subprocesses intact.
            with mock.patch.object(visualize.sys, "prefix", str(root / "empty-env")), \
                    mock.patch.object(visualize.shutil, "which", return_value=None), \
                    mock.patch.dict(visualize.os.environ, {"PATH": ""}):
                with self.assertRaises(FileNotFoundError):
                    visualize._media_binary("ffprobe")
                self.assertTrue(Path(visualize._media_binary("ffmpeg")).is_file())
                self.assertAlmostEqual(visualize._get_fps(str(clip)), fps)
                actual_width, actual_height, duration = visualize._get_video_info(str(clip))
                self.assertEqual((actual_width, actual_height), (width, height))
                self.assertAlmostEqual(duration, frame_count / fps, places=2)

                visualize.make_composite_compare_videos(str(source), str(output))
                composite = output / "motion_composite.mp4"
                self.assertTrue(composite.is_file())
                self.assertGreater(composite.stat().st_size, 0)
                result_width, result_height, result_duration = visualize._get_video_info(
                    str(composite)
                )
                self.assertEqual(result_width, width)
                self.assertGreater(result_height, height)
                self.assertAlmostEqual(result_duration, duration, places=2)
                self.assertAlmostEqual(visualize._get_fps(str(composite)), fps)
                reader = imageio_ffmpeg.read_frames(str(composite))
                try:
                    next(reader)
                    self.assertEqual(len(next(reader)), result_width * result_height * 3)
                finally:
                    reader.close()


if __name__ == "__main__":
    unittest.main()

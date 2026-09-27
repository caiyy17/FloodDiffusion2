"""Keep EGL driver discovery fallback scoped to unconfigured NVIDIA hosts."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import unittest
from unittest import mock

from visualization.tools import egl_setup


class EGLSetupTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(egl_setup.sys, "platform", "linux"))
        stack.enter_context(mock.patch.dict(os.environ, {
            "PYOPENGL_PLATFORM": "egl", "EGL_DEVICE_ID": "3"
        }, clear=True))
        stack.enter_context(mock.patch.object(Path, "glob", side_effect=lambda _: iter(())))
        self.driver = stack.enter_context(mock.patch.object(egl_setup.ctypes, "CDLL"))

    def test_missing_vendor_files_use_bundled_description(self):
        egl_setup.configure_egl_vendor()
        vendor = Path(os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"])
        self.assertEqual(json.loads(vendor.read_text()), {
            "file_format_version": "1.0.0",
            "ICD": {"library_path": "libEGL_nvidia.so.0"},
        })
        self.driver.assert_called_once_with("libEGL_nvidia.so.0")
        self.assertEqual(os.environ["EGL_DEVICE_ID"], "3")

    def test_explicit_driver_and_platform_settings_are_preserved(self):
        cases = [
            {"PYOPENGL_PLATFORM": "osmesa"},
            {"__EGL_VENDOR_LIBRARY_FILENAMES": "/custom/mesa.json"},
            {"__EGL_VENDOR_LIBRARY_DIRS": "/custom/vendors"},
            {"__EGL_VENDOR_LIBRARY_DIRS": ""},
            {"EGL_PLATFORM": "surfaceless"},
        ]
        for settings in cases:
            with self.subTest(settings=settings), mock.patch.dict(os.environ, settings):
                before = dict(os.environ)
                egl_setup.configure_egl_vendor()
                self.assertEqual(dict(os.environ), before)
        self.driver.assert_not_called()

    def test_installed_vendor_configuration_is_preserved(self):
        with mock.patch.object(Path, "glob", return_value=iter([Path("50_mesa.json")])):
            egl_setup.configure_egl_vendor()
        self.driver.assert_not_called()
        self.assertNotIn("__EGL_VENDOR_LIBRARY_FILENAMES", os.environ)

    def test_absent_nvidia_driver_does_not_change_discovery(self):
        self.driver.side_effect = OSError("NVIDIA EGL unavailable")
        egl_setup.configure_egl_vendor()
        self.assertNotIn("__EGL_VENDOR_LIBRARY_FILENAMES", os.environ)

    def test_non_linux_does_not_change_discovery(self):
        with mock.patch.object(egl_setup.sys, "platform", "win32"):
            egl_setup.configure_egl_vendor()
        self.driver.assert_not_called()
        self.assertNotIn("__EGL_VENDOR_LIBRARY_FILENAMES", os.environ)


if __name__ == "__main__":
    unittest.main()

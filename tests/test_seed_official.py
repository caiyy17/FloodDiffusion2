"""Protocol fixtures for official SEED captions, renamed moves and crop windows."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('seed_generate', Path(__file__).resolve().parents[1] / 'tools/seed_official/generate.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class OfficialCases(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.suite, self.data = root / 'suite', root / 'data'
        case = self.suite / 'case001'
        case.mkdir(parents=True)
        (case / 'meta.json').write_text(json.dumps({'duration': 1.2, 'seed': 23, 'text': 'A person jumps.'}))
        (case / 'seed_motion.json').write_text(json.dumps({'move_name': 'official_name', 'crop_start_frame_index': 7}))
        (self.data / 'texts').mkdir(parents=True)
        (self.data / 'texts/renamed.txt').write_text('first\nsecond\nthird\nA person jumps.#annotation\n')
        (self.data / 'new_joint_vecs_uni').mkdir()
        self.motion = np.arange(80 * 138, dtype=np.float32).reshape(80, 138)
        np.save(self.data / 'new_joint_vecs_uni/renamed.npy', self.motion)
        self.mapping = {'official_name': 'renamed'}

    def test_official_and_fourth_caption_match(self):
        a = adapter.load_cases(self.suite, self.data, self.mapping, False, -1)[0]
        b = adapter.load_cases(self.suite, self.data, self.mapping, True, 3)[0]
        self.assertEqual(a['text'], b['text'])
        self.assertEqual(a['seed'], 23)
        self.assertEqual(a['frames'], 36)
        np.testing.assert_array_equal(b['position'], self.motion[7:43, :3])

    def test_missing_crop_is_an_error_not_a_skipped_case(self):
        np.save(self.data / 'new_joint_vecs_uni/renamed.npy', self.motion[:40])
        with self.assertRaisesRegex(ValueError, 'unavailable official crop'):
            adapter.load_cases(self.suite, self.data, self.mapping, True, -1)

    def test_text_only_needs_no_motion_store(self):
        missing = self.data / 'does_not_exist'
        case = adapter.load_cases(self.suite, missing, {}, False, -1)[0]
        self.assertEqual(case['text'], 'A person jumps.')
        self.assertIsNone(case['position'])


if __name__ == '__main__':
    unittest.main()

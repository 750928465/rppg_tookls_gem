"""Version-policy tests; these do not pretend to load TorchAudio or a CUDA driver."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'phase_shifted_env_check', Path(__file__).resolve().parents[1] / 'tools/check_phase_shifted_env.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class AudioCompatibilityTests(unittest.TestCase):
    def test_stable_abi_accepts_newer_torch(self):
        for version in ('2.11.0+cu130', '2.12.0+cu130', '2.14.1+cu130'):
            with self.subTest(torch=version):
                checker.check_audio_compatibility(version, '2.11.0+cu130')

    def test_legacy_matching_pair_still_accepted(self):
        checker.check_audio_compatibility('2.8.0+cu128', '2.8.0+cu128')

    def test_incompatible_pairs_rejected(self):
        for torch_version, audio_version in [('2.10.0', '2.11.0'), ('2.14.1', '2.8.0'), ('2.8.0', '2.7.1')]:
            with self.subTest(torch=torch_version, audio=audio_version):
                with self.assertRaises(RuntimeError):
                    checker.check_audio_compatibility(torch_version, audio_version)


if __name__ == '__main__':
    unittest.main()

"""Cache-output location safety checks."""
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH=Path(__file__).resolve().parents[1]/"scripts/build_cache.py"
SPEC=importlib.util.spec_from_file_location("build_cache",MODULE_PATH)
build_cache=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_cache)


class CacheLocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)/"repo"
        self.root.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def result(self,returncode,stderr=""):
        return subprocess.CompletedProcess([],returncode,"",stderr)

    def test_ignored_path_inside_checkout_passes(self):
        cache=self.root/"Datasets/otf-cache/v1"
        with patch.object(build_cache.subprocess,"run",return_value=self.result(0)) as run:
            build_cache.validate_cache_location(self.root,cache)
        run.assert_called_once_with(["git","check-ignore","-q","--",str(cache.resolve())],
                                    cwd=self.root.resolve(),check=False,
                                    stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)

    def test_not_ignored_path_inside_checkout_fails(self):
        cache=self.root/"tracked-cache"
        with patch.object(build_cache.subprocess,"run",return_value=self.result(1)):
            with self.assertRaisesRegex(ValueError,"must be ignored by git"):
                build_cache.validate_cache_location(self.root,cache)

    def test_path_outside_checkout_passes_without_git(self):
        outside=Path(self.temporary.name)/"outside-cache"
        with patch.object(build_cache.subprocess,"run",side_effect=AssertionError("git called")):
            build_cache.validate_cache_location(self.root,outside)

    def test_missing_git_refuses_inside_checkout(self):
        with patch.object(build_cache.subprocess,"run",side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError,"git is unavailable"):
                build_cache.validate_cache_location(self.root,self.root/"ignored-cache")


if __name__=="__main__":
    unittest.main()

import io
import os
import shutil
import stat
import tempfile
import unittest
from contextlib import redirect_stdout

from language_pipes.util import clone_model

class TestCloneModel(unittest.TestCase):
    @unittest.skip("Skip successful download normally")
    def test_downloads_model_to_data_subdirectory(self):
        """clone_model should download model files to model_dir/data/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "test_model")
            
            clone_model("facebook/opt-125m", model_dir)
            
            # Verify the data directory was created
            data_dir = os.path.join(model_dir, "data")
            self.assertTrue(os.path.isdir(data_dir))
            
            # Verify essential model files exist
            self.assertTrue(os.path.exists(os.path.join(data_dir, "config.json")))
            shutil.rmtree(model_dir)

    def test_repository_not_found(self):
        """When model doesn't exist, model_dir should be removed and exit(1) called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "test_model")
            # Pre-create the directory to simulate partial state
            os.makedirs(model_dir)
            marker_file = os.path.join(model_dir, "marker.txt")
            with open(marker_file, "w") as f:
                f.write("partial download")
            
            self.assertTrue(os.path.exists(model_dir))
            
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                with self.assertRaises(SystemExit) as ctx:
                    clone_model("this-repo-definitely-does-not-exist-12345", model_dir)
            
            self.assertEqual(ctx.exception.code, 1)
            self.assertFalse(os.path.exists(model_dir))
            
            output = captured_output.getvalue()
            self.assertIn("not found on HuggingFace Hub", output)

    def test_gated_repo(self):
        """When model is gated and no token provided, model_dir should be removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "test_model")
            os.makedirs(model_dir)
            marker_file = os.path.join(model_dir, "marker.txt")
            with open(marker_file, "w") as f:
                f.write("partial download")
            
            self.assertTrue(os.path.exists(model_dir))
            
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                with self.assertRaises(SystemExit) as ctx:
                    # meta-llama/Llama-2-7b is a gated model requiring agreement
                    clone_model("meta-llama/Llama-2-7b", model_dir, token=None)
            
            self.assertEqual(ctx.exception.code, 1)
            self.assertFalse(os.path.exists(model_dir))
            
            output = captured_output.getvalue()
            self.assertIn("is gated", output)
            self.assertIn("huggingface_token", output)

    def test_permission_error(self):
        """When download fails due to permissions, error message should include exception details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a read-only directory as model_dir
            model_dir = os.path.join(tmpdir, "readonly_model")
            os.makedirs(model_dir)
            os.chmod(model_dir, stat.S_IRUSR | stat.S_IXUSR)  # read + execute only
            
            captured_output = io.StringIO()
            try:
                with redirect_stdout(captured_output):
                    with self.assertRaises(SystemExit) as ctx:
                        clone_model("facebook/opt-125m", model_dir)
                
                self.assertEqual(ctx.exception.code, 1)
                output = captured_output.getvalue()
                # Verify the error message contains useful information
                self.assertIn("Unexpected error", output)
                self.assertIn("Permission denied", output)
            finally:
                # Restore write permission so cleanup can succeed (if dir still exists)
                if os.path.exists(model_dir):
                    os.chmod(model_dir, stat.S_IRWXU)

if __name__ == "__main__":
    unittest.main()

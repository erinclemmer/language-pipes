import os
import sys
import toml
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.cli import main, build_parser, VERSION

CONFIG_PATH = "_tmp_config.toml"
KEY_PATH = "_tmp_network.key"

class KeygenTests(unittest.TestCase):
    def setUp(self):
        """Clean up any existing test files"""
        if os.path.exists(KEY_PATH):
            os.remove(KEY_PATH)

    def tearDown(self):
        """Clean up test files after each test"""
        if os.path.exists(KEY_PATH):
            os.remove(KEY_PATH)

    def test_keygen_creates_file(self):
        """keygen should create a key file at the specified path"""
        self.assertFalse(os.path.exists(KEY_PATH))
        main(["keygen", KEY_PATH])
        self.assertTrue(os.path.exists(KEY_PATH))

    def test_keygen_file_not_empty(self):
        """Generated key file should not be empty"""
        main(["keygen", KEY_PATH])
        with open(KEY_PATH, 'rb') as f:
            content = f.read()
        self.assertGreater(len(content), 0)

    def test_keygen_file_is_binary(self):
        """Generated key should be binary data (AES key)"""
        main(["keygen", KEY_PATH])
        with open(KEY_PATH, 'rb') as f:
            content = f.read()
        self.assertIn(len(content), [16, 24, 32, 64])

    def test_keygen_overwrites_existing(self):
        """keygen should overwrite an existing file"""
        with open(KEY_PATH, 'wb') as f:
            f.write(b'old content')
        
        main(["keygen", KEY_PATH])
        
        with open(KEY_PATH, 'rb') as f:
            content = f.read()
        self.assertNotEqual(content, b'old content')

    def test_keygen_different_keys(self):
        """keygen should generate different keys each time"""
        main(["keygen", KEY_PATH])
        with open(KEY_PATH, 'rb') as f:
            key1 = f.read()
        
        main(["keygen", KEY_PATH])
        with open(KEY_PATH, 'rb') as f:
            key2 = f.read()
        
        self.assertNotEqual(key1, key2)


class VersionTests(unittest.TestCase):
    def test_version_flag_short(self):
        """"-v" flag should trigger version output"""
        with self.assertRaises(SystemExit) as cm:
            main(["-v"])
        self.assertEqual(cm.exception.code, 0)

    def test_version_flag_long(self):
        """"--version" flag should trigger version output"""
        with self.assertRaises(SystemExit) as cm:
            main(["--version"])
        self.assertEqual(cm.exception.code, 0)

    def test_version_constant_exists(self):
        """VERSION constant should be defined"""
        self.assertIsNotNone(VERSION)
        self.assertIsInstance(VERSION, str)

    def test_version_format(self):
        """VERSION should be in semver format (x.y.z)"""
        parts = VERSION.split('.')
        self.assertGreaterEqual(len(parts), 2)
        # Each part should be numeric
        for part in parts:
            self.assertTrue(part.isdigit(), f"Version part '{part}' is not numeric")

class ServeConfigTests(unittest.TestCase):
    def setUp(self):
        """Clean up any existing test files"""
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)

    def tearDown(self):
        """Clean up test files after each test"""
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)

    def test_config_file_not_exist(self):
        """serve should fail if config file doesn't exist"""
        with self.assertRaises((FileNotFoundError, SystemExit)):
            main(["serve", "--config", "nonexistent.toml"])

    def test_config_file_empty(self):
        """serve should fail with empty config file"""
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write('')
        with self.assertRaises(SystemExit):
            main(["serve", "--config", CONFIG_PATH])

    def test_config_no_node_id(self):
        """serve should fail without node_id"""
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            toml.dump({
                "hosted_models": [{
                    "id": "meta-llama/Llama-3.2-1B",
                    "device": "cpu",
                    "max_memory": 5
                }]
            }, f)
        with self.assertRaises(SystemExit):
            main(["serve", "--config", CONFIG_PATH])

    def test_config_no_hosted_models(self):
        """serve should fail without hosted_models"""
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            toml.dump({
                "node_id": "node-1"
            }, f)
        with self.assertRaises(SystemExit):
            main(["serve", "--config", CONFIG_PATH])

    def test_hosted_models_cli_format(self):
        """serve should accept Docker-style key=value hosted-models"""
        parser = build_parser()
        args = parser.parse_args([
            "serve",
            "--node-id", "node-1",
            "--hosted-models", "id=Qwen/Qwen3-1.7B,device=cpu,memory=4"
        ])
        self.assertEqual(args.node_id, "node-1")
        self.assertEqual(args.hosted_models, ["id=Qwen/Qwen3-1.7B,device=cpu,memory=4"])

    def test_hosted_models_multiple(self):
        """serve should accept multiple hosted-models"""
        parser = build_parser()
        args = parser.parse_args([
            "serve",
            "--node-id", "node-1",
            "--hosted-models",
            "id=Qwen/Qwen3-1.7B,device=cpu,memory=4",
            "id=meta-llama/Llama-3.2-1B,device=cuda:0,memory=8"
        ])
        self.assertEqual(len(args.hosted_models), 2)


if __name__ == '__main__':
    unittest.main()

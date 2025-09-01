import os
import sys
import toml
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from language_pipes.cli import main 

CONFIG_PATH = "_tmp_config.toml"

class CliTests(unittest.TestCase):
    def test_empty(self):
        try:
            main([])
            self.fail("Should not succeed with no arguments")
        except:
            pass

    def test_config_file_not_exist(self):
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)
        try:
            main(["run", "--config", CONFIG_PATH])
            self.fail("Should not succeed with non existant config file")
        except:
            pass

    def test_config_file_empty(self):
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write('')
        try:
            main(["run", "--config", CONFIG_PATH])
            self.fail("Should not succeed with empty file")
        except:
            pass

    def test_config_min(self):
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            toml.dump({
                "node_id": "node-1",
                "hosted_models": [{
                    "id": "meta-llama",
                    "device": "cpu",
                    "max_memory": 5
                }]
            }, f)
        main(["run", "--config", CONFIG_PATH])
    
    def test_config_file_no_id(self):
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            toml.dump({
                "hosted_models": [{
                    "id": "meta-llama",
                    "device": "cpu",
                    "max_memory": 5
                }]
            }, f)
        try:
            main(["run", "--config", CONFIG_PATH])
            self.fail("Node ID should be neccessary to start program")
        except:
            pass

    def test_config_file_no_hosted(self):
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            toml.dump({
                "node_id": "node-1"
            }, f)
        try:
            main(["run", "--config", CONFIG_PATH])
            self.fail("hosted_models should be necessary to start program")
        except:
            pass

    def test_min_flags(self):
        main(["run", "--node-id", "node-1", "--hosted-models", "meta-llama/Llama3.2-1B-Instruct:cpu:5"])

if __name__ == '__main__':
    unittest.main()
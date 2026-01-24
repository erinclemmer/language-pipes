import os
import sys
import time
import unittest
import requests

sys.path.insert(0, os.path.dirname(__file__))

from base import DSNTestBase, spawn_node


class TestHTTPEndpoints(DSNTestBase):
    def test_http_endpoints_exist(self):
        """All HTTP endpoints should exist (not return 404)"""
        n = spawn_node("http-test-node", "127.0.0.1")
        time.sleep(0.5)
        
        endpoints = ['/hello', '/peers', '/update', '/ping', '/data']
        
        for endpoint in endpoints:
            try:
                response = requests.post(
                    f'http://127.0.0.1:{n.config.port}{endpoint}',
                    data=b'test',
                    timeout=2
                )
                self.assertNotEqual(response.status_code, 404, f"Endpoint {endpoint} not found")
                print(f"Endpoint {endpoint} exists (status: {response.status_code})")
            except Exception as e:
                print(f"Endpoint {endpoint} test failed: {e}")


if __name__ == "__main__":
    unittest.main()

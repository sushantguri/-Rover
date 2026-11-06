"""
Unit tests for FastAPI Microservice endpoints.
"""
import io
import unittest
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

from api import app


class TestFastApiEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_and_classes(self):
        """Verifies root status and classes metadata."""
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)
        self.assertEqual(r_root.json()["status"], "ONLINE")

        r_classes = self.client.get("/v1/classes")
        self.assertEqual(r_classes.status_code, 200)
        self.assertEqual(len(r_classes.json()["classes"]), 4)

    def test_segment_endpoint(self):
        """Verifies /v1/segment accepts image upload and returns telemetry."""
        img = Image.fromarray(np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("test_mars.jpg", buf, "image/jpeg")}
        resp = self.client.post("/v1/segment?resolution=256&include_base64_overlays=true", files=files)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("telemetry", data)
        self.assertIn("overlays", data)
        self.assertIn("multiclass_terrain", data["overlays"])
        self.assertIn("traversability_index", data["telemetry"])


if __name__ == "__main__":
    unittest.main()

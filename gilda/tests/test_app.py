"""Test the API in the Gilda app."""

import unittest

from gilda.app.app import app


class TestApp(unittest.TestCase):
    """A test case for the Gilda Flask application."""

    def test_post_grounding(self):
        """Test the POST response with text."""
        with app.test_client() as client:
            res = client.post(f"/ground", json={"text": "AKT"})
            self.assertIsNotNone(res)
            self.assertIsNotNone(res.json)
            self.assertIsInstance(res.json, list)

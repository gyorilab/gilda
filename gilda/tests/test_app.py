"""Test the API in the Gilda app."""

import unittest

from gilda.app.app import app


class TestApp(unittest.TestCase):
    """A test case for the Gilda Flask application."""

    def test_post_grounding(self):
        """Test the POST response with text."""
        with app.test_client() as client:
            res = client.post("/ground")
            self.assertIn("message", res.json)

            res = client.post("/ground", json={"text": "AKT1"})
            self.assertIsInstance(res.json, list)
            self.assert_found(res.json, "HGNC", "391")

            res = client.post("/ground", json={"text": "AKT1", "organisms": ["9606", "10090"]})
            self.assertIsInstance(res.json, list)
            self.assert_found(res.json, "HGNC", "391")
            self.assert_found(res.json, "MGI", "87986")

    def assert_found(self, matches, prefix: str, identifier: str) -> None:
        match_curies = {
            (match['term']['db'], match['term']['id'])
            for match in matches
        }
        self.assertIn((prefix, identifier), match_curies)

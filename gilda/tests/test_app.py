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

            res = client.post("/ground", json={"text": "Raf1"})
            self.assertIsInstance(res.json, list)
            self.assert_found(res.json, "HGNC", "9829")

            res = client.post("/ground", json={"text": "Raf1", "organisms": ["9606", "10090"]})
            self.assertIsInstance(res.json, list)
            self.assert_found(res.json, "HGNC", "9829")

            res = client.post("/ground", json={"text": "Raf1", "organisms": ["10090", "9606"]})
            self.assertIsInstance(res.json, list)
            self.assert_found(res.json, "UP", "Q99N57")

    def test_get_names(self):
        with app.test_client() as client:
            res = client.post('/names', json={"db": "FPLX", "id": "ERK"})
            self.assertIsInstance(res.json, list)
            self.assertIn('ERK 1/2', res.json)

    def test_get_models(self):
        with app.test_client() as client:
            res = client.get('/models')
            self.assertIsInstance(res.json, list)
            self.assertIn('ABC1', res.json)

    def assert_found(self, matches, prefix: str, identifier: str) -> None:
        match_curies = {
            (match['term']['db'], match['term']['id'])
            for match in matches
        }
        self.assertIn((prefix, identifier), match_curies)

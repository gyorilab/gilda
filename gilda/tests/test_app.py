"""Test the API in the Gilda app."""

import unittest
from typing import ClassVar

import flask

from gilda.app.app import get_app


class TestApp(unittest.TestCase):
    """A test case for the Gilda Flask application."""

    app: ClassVar[flask.Flask]

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_get_home(self):
        """Test the GET response on the home page."""
        with self.app.test_client() as client:
            res = client.get("/?text=Raf1")
            self.assertNotEqual(
                302,
                res.status_code,
                msg="Should not receive a redirect, this probably means the UI isn't mounted properly",
            )
            self.assert_raf1_ui(res)

    def test_post_home(self):
        """Test the POST response on the home page."""
        with self.app.test_client() as client:
            res = client.post("/", json={"text": "Raf1"})
            self.assertNotEqual(
                302,
                res.status_code,
                msg="Should not receive a redirect, this probably means the UI isn't mounted properly",
            )
            self.assert_raf1_ui(res)

    def assert_raf1_ui(self, res) -> None:
        text = res.data.decode()
        self.assertEqual(200, res.status_code)
        self.assertIn('RAF1', text)
        self.assertIn('0.9998', text)
        self.assertIn('RNASE3', text)
        self.assertIn('0.5024', text)

    def test_post_grounding(self):
        """Test the POST response with text."""
        with self.app.test_client() as client:
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
        with self.app.test_client() as client:
            res = client.post('/names', json={"db": "FPLX", "id": "ERK"})
            self.assertIsInstance(res.json, list)
            self.assertIn('ERK 1/2', res.json)

    def test_get_models(self):
        with self.app.test_client() as client:
            res = client.get('/models')
            self.assertIsInstance(res.json, list)
            self.assertIn('ABC1', res.json)

    def assert_found(self, matches, prefix: str, identifier: str) -> None:
        match_curies = {
            (match['term']['db'], match['term']['id'])
            for match in matches
        }
        self.assertIn((prefix, identifier), match_curies)

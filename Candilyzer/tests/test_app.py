import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "app.py"


class AppSmokeTests(unittest.TestCase):
    def test_empty_and_sample_states_render_without_exceptions(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(
            any("Candidate evaluation" in block.value for block in app.markdown)
        )

        app.button[0].click().run(timeout=20)

        self.assertFalse(app.exception)
        self.assertTrue(any("Evidence review" in block.value for block in app.markdown))
        self.assertTrue(any(tab.label == "Amina Karimova" for tab in app.tabs))


if __name__ == "__main__":
    unittest.main()

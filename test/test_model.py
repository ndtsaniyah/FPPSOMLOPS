import pytest
from app.utils import your_labeling_function

def test_labeling_function():
    input_data = "This is a sample input text."
    expected_output = {"label": "Expected label"}  # Sesuaikan dengan output yang diharapkan
    assert your_labeling_function(input_data) == expected_output

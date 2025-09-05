import pytest
from torchstore.logging import init_logging

def test_main(file):
    init_logging()
    pytest.main([file])

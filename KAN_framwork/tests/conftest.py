import pytest
from kangnn import seed_everything


@pytest.fixture(autouse=True)
def deterministic():
    seed_everything(17)

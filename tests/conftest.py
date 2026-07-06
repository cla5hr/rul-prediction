import pytest

from rul_service.artifacts import load_bundle
from rul_service.config import PROJECT_DIR, Config
from rul_service.predict import Predictor


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory) -> Config:
    """A fast config (1 epoch) that trains real artifacts into a temp dir."""
    art = tmp_path_factory.mktemp("artifacts")
    return Config(epochs=1, model_type="lstm", artifacts_dir=art, data_dir=PROJECT_DIR / "data")


@pytest.fixture(scope="session")
def trained(tiny_cfg) -> Config:
    from rul_service.train import train

    train(tiny_cfg, verbose=False)
    return tiny_cfg


@pytest.fixture(scope="session")
def predictor(trained) -> Predictor:
    bundle = load_bundle(trained.artifacts_dir)
    return Predictor(bundle, cfg=trained)

"""
Tests for the config module
"""
from pathlib import Path

from tno.fl.protocols.logistic_regression.config import Config, Party


def test_party_eq() -> None:
    """
    Test correctness of Party.__eq__ implementation
    """
    party_1 = Party(name="Alfred", address="1.2.3.4", port=80, train_data_path=Path())
    party_2 = Party(name="Bert", address="1.2.3.4", port=80, train_data_path=Path())
    assert party_1 == party_2

    party_1 = Party(name="Alfred", address="1.2.3.2", port=80, train_data_path=Path())
    party_2 = Party(name="Alfred", address="1.2.3.4", port=80, train_data_path=Path())
    assert party_1 != party_2

    party_1 = Party(name="Alfred", address="1.2.3.4", port=81, train_data_path=Path())
    party_2 = Party(name="Alfred", address="1.2.3.4", port=80, train_data_path=Path())
    assert party_1 != party_2


def test_config_parse() -> None:
    """
    Test parsing a configuration file
    """

    # Parse the config file
    config = Config.from_file(Path("examples/iris/iris.ini"))

    # Test correctness
    correct_config = Config(
        clients=[
            Party(
                name="Alice",
                address="localhost",
                port=8001,
                train_data_path=Path("data_alice.csv"),
            ),
            Party(
                name="Bob",
                address="localhost",
                port=8002,
                train_data_path=Path("data_bob.csv"),
            ),
        ],
        server=Party(
            name="Server",
            address="localhost",
            port=8000,
            train_data_path=Path("."),
        ),
        data_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        target_column="is_setosa",
        intercept=True,
        n_epochs=50,
        learning_rate=None,
        hessian=True,
    )
    assert config == correct_config

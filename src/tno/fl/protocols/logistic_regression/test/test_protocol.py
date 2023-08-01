"""
Test to run entire protocol
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from tno.mpc.communication import Pool

from tno.fl.protocols.logistic_regression.client import Client
from tno.fl.protocols.logistic_regression.config import Config
from tno.fl.protocols.logistic_regression.server import Server

CONFIG_TEXT = """
[Experiment]
data_columns=sepal_length,sepal_width,petal_length,petal_width
target_column=is_setosa
intercept=True
n_epochs=50
learning_rate=hessian

[Parties]
clients=local1,local2
server=Server

[Server]
address=localhost
port=8000

[local1]
address=localhost
port=8001
train_data=examples/iris/data_alice.csv

[local2]
address=localhost
port=8002
train_data=examples/iris/data_bob.csv
"""


@pytest.mark.asyncio
async def test_protocol(
    tmp_path: Path, http_pool_trio: tuple[Pool, Pool, Pool]
) -> None:
    """
    Run the Cox regression with the Rotterdam example

    :param tmp_path: temporary path fixture
    :param http_pool_trio: communication pool fixture for 3 parties
    """
    # Set up the config file
    config_filename = tmp_path / "config.ini"
    with open(config_filename, "w", encoding="UTF-8") as text_file:
        text_file.write(CONFIG_TEXT)

    # Run the protocol
    config = Config.from_file(config_filename)
    server = Server(config)
    client1 = Client(config, "local1")
    client2 = Client(config, "local2")

    server.pool = http_pool_trio[0]
    client1.pool = http_pool_trio[1]
    client2.pool = http_pool_trio[2]
    client1.SERVER_ID = "local0"
    client2.SERVER_ID = "local0"

    models = await asyncio.gather(server.run(), client1.run(), client2.run())
    assert (models[0] == models[1]).all()
    assert (models[0] == models[2]).all()
    print(models[0])
    assert np.isclose(
        models[0],
        np.expand_dims(
            [-22.44124982, 18.50550676, 14.90706874, -34.66446449, -46.61405812],
            -1,
        ),
        rtol=1e-05,
        atol=1e-08,
        equal_nan=False,
    ).all()

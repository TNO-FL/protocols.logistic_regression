"""
Test to run entire protocol
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from tno.mpc.communication import Pool

from tno.fl.protocols.logistic_regression.client import Client, ModelType
from tno.fl.protocols.logistic_regression.server import Server


@pytest.mark.asyncio
async def test_protocol(http_pool_trio: tuple[Pool, Pool, Pool]) -> None:
    """
    Run the Cox regression with the Rotterdam example

    :param http_pool_trio: communication pool fixture for 3 parties
    """
    # Load the data
    csv_data_alice = pd.read_csv("examples/iris/data_alice.csv")
    data_alice = csv_data_alice[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].to_numpy()
    target_alice = csv_data_alice["is_setosa"].to_numpy()

    csv_data_bob = pd.read_csv("examples/iris/data_bob.csv")
    data_bob = csv_data_bob[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].to_numpy()
    target_bob = csv_data_bob["is_setosa"].to_numpy()

    # Run the protocol
    server = Server(http_pool_trio[0], max_iter=10)
    client1 = Client(http_pool_trio[1], max_iter=10, server_name="local0")
    client2 = Client(http_pool_trio[2], max_iter=10, server_name="local0")

    models: list[ModelType | None] = await asyncio.gather(
        *[
            server.run(),
            client1.run(data_alice, target_alice),
            client2.run(data_bob, target_bob),
        ]
    )

    # Tests
    assert models[1] is not None
    assert models[2] is not None
    assert np.array_equal(models[1], models[2])
    assert np.isclose(
        models[1],
        np.array([0.90271, 4.91107, -5.17815, -6.97310]),
        rtol=1e-05,
        atol=1e-05,
        equal_nan=False,
    ).all()

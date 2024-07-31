"""
This module runs the logistic regression protocol on an example data set.
By running the script three times with command line argument 'server', 'alice'
and 'bob' respectively, you can get a demonstration of how it works.
"""

import asyncio
import logging
import sys

import pandas as pd

from tno.mpc.communication import Pool

from tno.fl.protocols.logistic_regression.client import Client
from tno.fl.protocols.logistic_regression.server import Server

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def run_client(name: str, port: int) -> None:
    # Create Pool
    pool = Pool()
    pool.add_http_server(addr="localhost", port=port)
    pool.add_http_client(name="server", addr="localhost", port=8080)
    # Get Data
    csv_data = pd.read_csv("data_" + name + ".csv")
    data = csv_data[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].to_numpy()
    target = csv_data["is_setosa"].to_numpy()
    # Create Client
    client = Client(pool, max_iter=50)
    logger.info(await client.run(data, target))
    await pool.shutdown()


async def run_server() -> None:
    # Create Pool
    pool = Pool()
    pool.add_http_server(addr="localhost", port=8080)
    pool.add_http_client(name="alice", addr="localhost", port=8081)
    pool.add_http_client(name="bob", addr="localhost", port=8082)
    # Create Client
    server = Server(pool, max_iter=50)
    await server.run()
    await pool.shutdown()


async def async_main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Player name must be provided.")
    if sys.argv[1].lower() == "server":
        await run_server()
    elif sys.argv[1].lower() == "alice":
        await run_client("alice", 8081)
    elif sys.argv[1].lower() == "bob":
        await run_client("bob", 8082)
    else:
        raise ValueError(
            "This player has not been implemented. Possible values are: server, alice, bob"
        )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_main())

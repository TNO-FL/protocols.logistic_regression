"""
Logistic regression client
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from tno.mpc.communication import Pool

from . import msg_ids
from .config import Config


class Client:
    """
    The client class, representing data owning clients in the learning process.

    :param SERVER_ID: The name for the server
    """

    SERVER_ID = "Server"

    def __init__(self, config: Config, client_name: str) -> None:
        """
        Initializes the client based on experiment config and the running client.
        This includes setting up communication channels and parsing the client data.

        :param config: The experiment configuration.
        :param client_name: The name of client running the application.
        """
        # Set the config
        self.config = config
        # Set client id
        self.client = self.config.get_client_by_name(client_name)
        # Create the communication pool
        self.pool = self.create_communication_pool()
        # Read the data from the file
        self.data, self.target = self.load_data()

    def create_communication_pool(self) -> Pool:
        """
        Create the communication pool with the client and the server.

        :return: The communication pool
        """
        pool = Pool()
        pool.add_http_server(addr=self.client.address, port=self.client.port)
        pool.add_http_client(
            name=self.SERVER_ID,
            addr=self.config.server.address,
            port=self.config.server.port,
        )
        return pool

    def load_data(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """
        Parse the data from the input data file and split in train data and target values.

        :return: A tuple containing a matrix of training data and a vector of target values.
        :raises FileNotFoundError: if the data file does not exist.
        """
        path = self.client.train_data_path
        if not path.exists():
            raise FileNotFoundError("The training data does not exist: ", path)
        csv_data: pd.DataFrame = pd.read_csv(path)
        return (
            csv_data[self.config.data_columns].to_numpy(),
            csv_data[self.config.target_column].to_numpy(),
        )

    async def preprocessing(self) -> None:
        """
        Perform preprocessing on the data. Adds an intercept column if needed according to config.
        """
        # Add column of ones for intercept terms (if applicable)
        if self.config.intercept:
            self.data = np.concatenate(
                (np.ones((self.data.shape[0], 1)), self.data), axis=1
            )

    async def send_weight_to_server(self) -> None:
        """
        Send the weights (number of rows) to the server.
        """
        await self.pool.send(self.SERVER_ID, len(self.data), msg_id=msg_ids.WEIGHT)

    def initial_model(self) -> npt.NDArray[np.float64]:
        """
        Computes the initial local model.

        :return: The initial local model.
        """
        return np.zeros((self.data.shape[1], 1))

    def compute_gradient(
        self, coefs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Compute the first-order gradient of the coefficients on the data.

        :param coefs: The coefficients at which to compute the gradient.
        :return: The gradient vector.
        """
        # Transform labels array to column vector
        target_vector = self.target[:, np.newaxis]
        # Compute predicted probabilities using sigmoid
        prob = 1 / (1 + np.exp(-np.dot(self.data, coefs)))
        # Compute gradient
        gradient: npt.NDArray[np.float64] = np.dot(self.data.T, (prob - target_vector))
        # Return gradient and hessian
        return gradient

    def compute_hessian(
        self, coefs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Compute the second-order derivative of the coefficients on the data.

        :param coefs: The coefficients at which to compute the gradient.
        :return: The hessian matrix.
        """
        # Compute predicted probabilities using sigmoid
        prob = 1 / (1 + np.exp(-np.dot(self.data, coefs)))
        # Compute diagonal matrix of weights
        weights = np.diag((prob * (1 - prob)).flatten())
        # Compute Hessian matrix
        hessian: npt.NDArray[np.float64] = np.dot(
            np.dot(self.data.T, weights), self.data
        )
        return hessian

    async def run(self) -> npt.NDArray[np.float64]:
        """
        Perform the learning process.

        :return: The resulting model.
        """
        # Preprocess the data
        await self.preprocessing()
        # The server needs the weight of each client (usually number of data points)
        await self.send_weight_to_server()
        # Initialize model
        model = self.initial_model()

        for _epoch in range(self.config.n_epochs):
            # Compute and send gradients and hessians
            gradient = self.compute_gradient(model)
            await self.pool.send(
                self.SERVER_ID, gradient, msg_id=msg_ids.LOCAL_GRADIENT
            )

            # Compute hessian if applicable
            if self.config.hessian:
                hessian = self.compute_hessian(model)
                await self.pool.send(
                    self.SERVER_ID, hessian, msg_id=msg_ids.LOCAL_HESSIAN
                )

            # Wait for and update coefficients
            model = await self.pool.recv(self.SERVER_ID, msg_id=msg_ids.COEFS)

        return model

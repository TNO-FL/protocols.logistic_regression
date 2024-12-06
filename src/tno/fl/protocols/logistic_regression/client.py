"""
Logistic regression client
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import scipy

from tno.mpc.communication import Pool

from tno.fl.protocols.logistic_regression.msg_ids import (
    COEFS,
    INIT_MODEL,
    LOCAL_HESSIAN,
    LOCAL_UPDATE,
    STANDARD_ERROR,
    WEIGHT,
)

logger = logging.getLogger(__name__)

DataType = npt.NDArray[np.float64]
TargetType = npt.NDArray[np.bool_]
ModelType = npt.NDArray[np.float64]
GradientType = npt.NDArray[np.float64]
HessianType = npt.NDArray[np.float64]
UpdateType = tuple[GradientType, HessianType]


class Client:
    """
    The client class, representing data owning clients in the learning process.
    """

    def __init__(
        self,
        pool: Pool,
        max_iter: int = 25,
        server_name: str = "server",
    ) -> None:
        """
        Initializes the client.

        :param pool: The communication pool.
        :param max_iter: The max number of epochs
        :param server_name: The name of the server
        """
        # Set the pool
        self.pool = pool
        # Set max number of epochs
        self.max_iter = max_iter
        # Set the server name
        self.server_name = server_name

    async def _compute_local_weight(self, data: DataType, target: TargetType) -> float:
        """
        Compute the local weight of a client. Defaults to the length of the data set.

        :param data: The data for the client
        :param target: The target data for the client
        :return: The weight of the client dataset
        """
        del target
        return len(data)

    async def _send_weight_to_server(self, weight: float) -> None:
        """
        Send the weights (number of rows) to the server.

        :param weight: The weight of the client
        """
        await self.pool.send(self.server_name, weight, msg_id=WEIGHT)

    async def _get_initial_model(self, data: DataType, target: TargetType) -> ModelType:
        """
        Computes the initial local model.

        :param data: The data for the client
        :param target: The target data for the client
        :return: The initial local model.
        """
        del target
        # Send local model to server
        local_model = np.zeros(data.shape[1])
        await self.pool.send(self.server_name, local_model, msg_id=INIT_MODEL)
        # Receive global initial model from server
        model = np.array(
            await self.pool.recv(self.server_name, msg_id=INIT_MODEL),
            dtype=np.float64,
        )
        return model

    def _compute_gradient(
        self, data: DataType, target: TargetType, model: ModelType
    ) -> GradientType:
        """
        Compute the first-order gradient of the coefficients on the data.

        :param data: The data set
        :param target: The target data
        :param model: The coefficients at which to compute the gradient.
        :return: The gradient vector.
        """
        # Transform labels array to column vector
        target_vector = target[:, np.newaxis]
        # Compute predicted probabilities using sigmoid
        prob = 1 / (1 + np.exp(-np.dot(data, model.reshape(-1, 1))))
        # Compute gradient
        gradient: GradientType = np.dot(data.T, (prob - target_vector))
        # Return gradient and hessian
        return gradient

    def _compute_hessian(self, data: DataType, model: ModelType) -> HessianType:
        """
        Compute the second-order derivative of the coefficients
        on the data.

        :param data: The data set
        :param model: The coefficients at which to compute the gradient.
        :return: The hessian matrix.
        """
        # Compute predicted probabilities using sigmoid
        prob = 1 / (1 + np.exp(-np.dot(data, model.reshape(-1, 1))))
        # Compute diagonal matrix of weights
        hessian: GradientType = np.dot(np.multiply(data, prob * (1 - prob)).T, data)
        return hessian

    def _compute_update(
        self, data: DataType, target: TargetType, model: ModelType
    ) -> UpdateType:
        """
        Compute the update for the client: the gradient and hessian

        :param data: The data set
        :param target: The target data
        :param model: The coefficients at which to compute the gradient.
        :return: The gradient vector.
        """
        gradient = self._compute_gradient(data, target, model)
        hessian = self._compute_hessian(data, model)
        return gradient, hessian

    async def _send_update_to_server(self, update: UpdateType) -> None:
        """
        Send the gradient to the server

        :param update: The gradient to be sent to the server
        """
        await self.pool.send(self.server_name, update, msg_id=LOCAL_UPDATE)

    async def _receive_model(self) -> ModelType:
        """
        Receive the updated model from server

        :return: The updated model
        """
        return np.array(
            await self.pool.recv(self.server_name, msg_id=COEFS),
            dtype=np.float64,
        )

    async def _run_epoch(
        self, data: DataType, target: TargetType, model: ModelType
    ) -> ModelType:
        """
        Perform one epoch.

        :param data: The data set
        :param target: The target data
        :param model: The model at the start of the epoch.
        :return: The updated model after the epoch.
        """
        # Compute the update and send to server
        update = self._compute_update(data, target, model)
        await self._send_update_to_server(update)

        # Wait for and update coefficients
        model = await self._receive_model()

        # Return the model
        return model

    async def run(self, data: DataType, target: TargetType) -> ModelType:
        """
        Perform the learning process.

        :param data: The training data for the client
        :param target: The target data for the client. Must be an array of boolean values.
        :return: The resulting model.
        """
        # The server needs the weight of each client (usually number of data points)
        local_weight = await self._compute_local_weight(data, target)
        await self._send_weight_to_server(local_weight)
        # Initialize model
        model = await self._get_initial_model(data, target)

        # Run the learning process
        for _epoch in range(self.max_iter):
            logger.info(f"Performing epoch {_epoch}/{self.max_iter}")
            model = await self._run_epoch(data, target, model)

        return model

    async def compute_standard_error(
        self, data: DataType, target: TargetType, model: ModelType
    ) -> npt.NDArray[np.float64]:
        """
        Compute the standard error for a model.

        :param data: The data set
        :param target: The target data
        :param model: The parameters for which to compute the standard error.
        :return: The standard error
        """
        # Compute Hessian
        hessian = self._compute_hessian(data, model)
        # Share weight and Hessian with server
        weight = await self._compute_local_weight(data, target)
        await self.pool.send(self.server_name, (weight, hessian), msg_id=LOCAL_HESSIAN)
        # Receive standard error
        return np.array(await self.pool.recv(self.server_name, msg_id=STANDARD_ERROR))

    async def compute_statistics(
        self, data: DataType, target: TargetType, model: ModelType
    ) -> list[dict[str, float]]:
        """
        Compute statistics for each coefficient: standard error, z-value and p-value.

        :param data: The data set
        :param target: The target data
        :param model: The model for which to compute the statistics
        :return: A list containing a dictionary for each covariate. The dictionary contains three values:
            'se' containing the standard error,
            'z' containing z-value (Wald statistic)
            'p' containing the p-value.
        """
        standard_errors = await self.compute_standard_error(data, target, model)
        z_values = model / standard_errors
        p_values = 2 * (1 - scipy.stats.norm.cdf(abs(z_values)))
        return [
            {"se": standard_errors[i], "z": z_values[i], "p": p_values[i]}
            for i in range(len(standard_errors))
        ]

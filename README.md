# TNO PET Lab - Federated Learning (FL) - Protocols - Logistic Regression

The TNO PET Lab consists of generic software components, procedures, and
functionalities developed and maintained on a regular basis to facilitate and
aid in the development of PET solutions. The lab is a cross-project initiative
allowing us to integrate and reuse previously developed PET functionalities to
boost the development of new protocols and solutions.

The package `tno.fl.protocols.logistic_regression` is part of the TNO Python
Toolbox.

Implementation of a Federated Learning scheme for Logistic Regression. This
library was designed to facilitate both developers that are new to cryptography
and developers that are more familiar with cryptography.

Supports:

- Any number of clients and one server.
- Horizontal fragmentation
- Binary classification (multi-class not yet supported)
- Both fixed learning rate or second-order methods (Hessian)

This software implementation was financed via EUREKA ITEA labeling under Project
reference number 20050.

_Limitations in (end-)use: the content of this software package may solely be
used for applications that comply with international export control laws.  
This implementation of cryptographic software has not been audited. Use at your
own risk._

## Documentation

Documentation of the tno.fl.protocols.logistic_regression package can be found
[here](https://docs.pet.tno.nl/fl/protocols/logistic_regression/0.2.2).

## Install

Easily install the tno.fl.protocols.logistic_regression package using pip:

```console
$ python -m pip install tno.fl.protocols.logistic_regression
```

If you wish to run the tests you can use:

```console
$ python -m pip install 'tno.fl.protocols.logistic_regression[tests]'
```

## Usage

This package uses federated learning for training a logistic regression model on
datasets that are distributed amongst several clients. Below is first a short
overview of federated learning in general and how this has been implemented in
this package. In the next section, a minimal working example is provided.

### Federated Learning

In Federated Learning, several clients, each with their own data, wish to fit a
model on their combined data. Each client computes a local update on their model
and sends this update to a central server. This server combines these updates,
updates the global model from this aggregated update and sends this new model
back to the clients. Then the process repeats: the clients compute the local
updates on this new model, send this to the server, which combines it and so on.
This is done until the server notices that the model has converged.

This package implements binary logistic regression. So each client has a data
set, that contains data and for each row a binary indicator. The goal is to
predict the binary indicator for new data. For example, the data could images of
cats and dogs and the binary indicator indicates whether it is a cat or a dog.
The goal of the logistic regression model, is to predict for new images whether
it contains a cat or a dog. More information on logistic regression is widely
available.

In the case of logistic regression, the updates the client compute consist of a
gradient. This model also implements a second-order derivative (Newton's
method).

### Implementation

The implementation of federated logistic regression consist of two classes with
the suggestive names `Client` and `Server`. Each client is an instance of
`Client` and the server is an instance of the `Server` class. These classes are
passed a configuration object and a name (unique identifier for the client).
Calling the `.run()` method on the objects, will perform the federated learning
and returns the resulting logistic regression model (numpy array).

All settings are defined in a configuration file. This file is a `.ini` file and
a template is given in the `config.ini` (in the repository). An example is also
shown below in the minimal example. Here is an overview of what must be in the
configuration.

The files contains a Parties section in which the names of all clients and the
name of the server are listed. Next we have a separate section for each client
and server, containing the IP-address and port on which it can be reached. The
clients also have a link to the location of the `.csv`-file containing the
data.  
The 'Experiment' section contains the experiment configuration. Most of the
fields are self-explanatory:

- **data_columns**: the columns in the csv which should be used for training.
- **target_column**: the target column in the csv (which should be predicted).
- **intercept**: whether an intercept column should be added.
- **n_epochs**: maximum number of epochs
- **learning_rate**: the learning rate (float) or 'hessian'. If this value is
  'hessian', a second-order derivative is used as learning rate (Newton's
  method).

_Note: At this moment, only csv-files are supported as input. Users can use
other file types or databases by overriding the `load_data()` method on the
clients._

#### Communication

This package relies on the `tno.mpc.communication` package, which is also part
of the PET lab. It is used for the communication amongst the server and the
clients. Since this package uses `asyncio` for asynchronous handling, this
federated learning package depends on it as well. For more information about
this, we refer to the
[tno.mpc.communication documentation](https://docs.pet.tno.nl/mpc/communication/)

### Example code

Below is a very minimal example of how to use the library. It consists of two
clients, Alice and Bob, who want to fit a model for recognizing the setosa iris
flower. Below is an excerpt from their data sets:

`data_alice.csv`

```csv
sepal_length,sepal_width,petal_length,petal_width,is_setosa
5.8,2.7,5.1,1.9,0
6.9,3.1,5.4,2.1,0
5,3.4,1.5,0.2,1
5.2,4.1,1.5,0.1,1
6.7,3.1,5.6,2.4,0
6.3,2.9,5.6,1.8,0
5.6,2.5,3.9,1.1,0
5.7,3.8,1.7,0.3,1
5.8,2.6,4,1.2,0
```

`data_bob.csv`

```csv
sepal_length,sepal_width,petal_length,petal_width,is_setosa
7.2,3,5.8,1.6,0
6.7,2.5,5.8,1.8,0
6,3.4,4.5,1.6,0
4.8,3.4,1.6,0.2,1
7.7,3.8,6.7,2.2,0
5.4,3.9,1.3,0.4,1
7.7,3,6.1,2.3,0
7.1,3,5.9,2.1,0
6.1,2.9,4.7,1.4,0
```

Next, we create a configuration file for this experiment.

`iris.ini`

```text
[Experiment]
data_columns=sepal_length,sepal_width,petal_length,petal_width
target_column=is_setosa
intercept=True
n_epochs=10
learning_rate=hessian

[Parties]
clients=Alice,Bob
server=Server

[Server]
address=localhost
port=8000

[Alice]
address=localhost
port=8001
train_data=data_alice.csv

[Bob]
address=localhost
port=8002
train_data=data_bob.csv
```

Finally, we create the code to run the federated learning algorithm:

`main.py`

```python
import asyncio
import sys
from pathlib import Path

from tno.fl.protocols.logistic_regression.client import Client
from tno.fl.protocols.logistic_regression.config import Config
from tno.fl.protocols.logistic_regression.server import Server


async def async_main() -> None:
    config = Config.from_file(Path("iris.ini"))
    if sys.argv[1].lower() == "server":
        server = Server(config)
        print(await server.run())
    elif sys.argv[1].lower() == "alice":
        client = Client(config, "Alice")
        print(await client.run())
    elif sys.argv[1].lower() == "bob":
        client = Client(config, "Bob")
        print(await client.run())
    else:
        raise ValueError(
            "This player has not been implemented. Possible values are: server, alice, bob"
        )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_main())
```

To run this script, call `main.py` from the folder where the data files and the
config file are located. As command line argument, pass it the name of the party
running the app: 'Alice', 'Bob', or 'Server'. To run in on a single computer,
run the following three command, each in a different terminal: Note that if a
client is started prior to the server, it will throw a ClientConnectorError.
Namely, the client tries to send a message to port the server, which has not
been opened yet. After starting the server, the error disappears.

```console
python main.py alice
python main.py bob
python main.py server
```

The output for the clients will be something similar to:

```console
>>> python main.py alice
2023-07-31 14:21:21,765 - tno.mpc.communication.httphandlers - INFO - Serving on localhost:8001
2023-07-31 14:21:21,780 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
2023-07-31 14:21:21,796 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
2023-07-31 14:21:21,811 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
2023-07-31 14:21:21,833 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
2023-07-31 14:21:21,833 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
2023-07-31 14:21:21,851 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
2023-07-31 14:21:21,867 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
2023-07-31 14:21:21,882 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
2023-07-31 14:21:21,898 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
2023-07-31 14:21:21,914 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8000
[[-2.907941249994596], [-0.1876483927585601], [7.728309577725918], [-6.938238886471739], [0.5467650097181181]]
```

We first see the client setting up the connection with the server. Then we have
ten rounds of training, as indicated in the configuration file. Finally, we
print the resulting model. We obtain the following coefficients for classifying
setosa irises:

| Parameter    | Coefficient         |
| ------------ | ------------------- |
| intercept    | -2.907941249994596  |
| sepal_length | -0.1876483927585601 |
| sepal_width  | 7.728309577725918   |
| petal_length | -6.938238886471739  |
| petal_width  | 0.5467650097181181  |

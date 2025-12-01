# Federated Learning - Semi-Descentralized Architecture

## Description
Federated Semi-Descentralized Learning Architecture implementation in python for Diabetes Dataset prediction using Multilayer Perceptron and communication between multiple nodes through TCP sockets.

## Requirements
For run the projects it is necessary to have the requirements.txt file.

## Docker contruction and raising
### Docker image construction
To build the Docker image run the command.

```bash
docker build -t federated-semidescentralized_image .
```

### Docker containers construction
To run multiple container it is necessary to have installed Docker compose in your machine. By default the docker-compose.yml in the project is configurated for create 4 nodes, but you can modify the project for create more nodes.

```bash
docker compose up
```

## Project Architecture

- **nodeC** Contains all the server node logic.
  -  **models** Contains .keras models.
  - **avg_models.py** Contains Aggregated models functions.
  - **connections.py** Defines all network logic connections.
  - **server.py** Handles the server-side logic.

- **nodex** Contains all the edge nodes logis.
  - **models_#** Contains .keras models.
  - **connections.py** Defines all network logic connections.
  - **model_build.py** Defines the model architecture and training functions.
  - **client.py** Handles the client-side logic.

- **diabetes_divided** Data of Diabetes diagnosis.
- **coordination.py** Coordinates the role nodes selection.
- **utils.py** Auxiliar functions
- **main.py** Starts the python project.
- **Dockerfile** File for Docker image building.
- **docker-compose.yaml** File for multiple Docker container raising.
- **requirements.txt** Python required libraries for the project dependencies.
- **metrics.sh** Bash script for get computer capabilities per node.

## Principal Configurations
The main configuration must be done in the main.py file and de docker-compose.yaml file where are located all the virtual envioremental declarations of the system.

In the main.py file are configured:

```PY
ROUNDS = 5
SUB_ROUNDS = 5
IP = "172.23.211.39"
BIND_PORT = int(os.getenv("BIND_PORT", 5000))
DOCKER_PORT = int(os.getenv("DOCKER_PORT", 5000))
NODE_ID = int(os.getenv("NODE_ID"))

PARAMS = {
    "hidden_layers": [(32, 0.4), (16, 0.3)],
    "activation": "relu",
    "optimizer": "adam"
}
```
### Variable description

- The **ROUND** variable determines the number of times the system select a new central node.
- The **SUB_ROUNDS** variable determines the number of training epochs per ROUND.
- **IP** refers to the IP adress of the current node.
- **BIND_PORT** is the port on which Docker will bind the application.
- **DOCKER_PORT** is the port that is mapped inside de Docker container.
- **NODE_ID** is a unique identifier for the node.
- **PARAMS** are the hyperparameters used for the neural network model construction.


## System Operation

The algorithm behind of the system consists of a Distributed system organized by N nodes which at the init of the process comunicate with each other establishing their respective roles through the next policy defined at coordination.py file:

```PY
ganador = max(nodos, key=lambda x: 0.5*(0.5 * x['net_up'] + 0.5 * x['net_down']) + 0.3*x['ram'] + 0.35*x['cpu_mhz'] + 0.2*int(x['gpu']) + 0.1*(1/int(x['id'])))
```
All nodes execute the metrics.sh script, which allows them to gather the information required to fulfill the role policy. A key element of this process is a formula that computes a value based on each node’s ID, helping resolve any discrepancies in assigned roles across nodes.

As a result, all nodes can independently determine which nodes hold specific roles, removing the need for redundant communication.

The server node then initializes the first round and waits for the remaining nodes to connect. Once all nodes have been identified, the server creates an untrained model and sends it to the clients. Each client receives the model and trains it using its local data. Afterward, the edge nodes send their best trained models for the sub-round back to the server, which is responsible for averaging them, completing the aggregation phase.

This process repeats for M sub-rounds before the coordination step is performed again to determine the next server node. The entire procedure can be stopped early before reaching the maximum number of rounds if convergence is detected or if the system’s performance worsens.

The stopping policy is implemented in the utils.py file: 

```PY
def checkConvergence(scores:list[list[float, float, float]], patience:int, threshold:float=0.01)->bool:
    if len(scores) < patience:
        return False

    recent_scores = scores[-patience]
    for node in range(len(scores[-1])):
      diff = recent_scores[node] - scores[-1][node]
      if diff > threshold:
          return False
      if diff < -threshold:
          return False

    return True
```

# Results
All models are saved in their corresponding models/ directories inside both the nodex and nodeC folders.
The best model can be found in nodeC/models/avg, where all aggregated models are stored.
To identify the best-performing one, use the scores.txt* file, which contains the evaluation metrics for each model.

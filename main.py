from nodeC.server import server
from nodex.client import client
from coordination import coordinate
from utils import checkConvergence
import os
import time

ROUNDS = 5
SUB_ROUNDS = 5

# TU LÓGICA DE IP FIJA
IP = "172.23.211.39"
BIND_PORT = int(os.getenv("BIND_PORT", 5000))
DOCKER_PORT = int(os.getenv("DOCKER_PORT", 5000))
NODE_ID = int(os.getenv("NODE_ID"))
NETWORK_ADDRESSES = [f"{IP}:{port.strip()}" for port in os.getenv("PORTS").split(",")]
DOCKER_ADDRESS = f"{IP}:{DOCKER_PORT}"
PEERS = [port for port in NETWORK_ADDRESSES if port != DOCKER_ADDRESS]


NCLIENTS = len(NETWORK_ADDRESSES)
NODE_DIR = f"node{NODE_ID}"

PARAMS = {
    "hidden_layers": [(32, 0.4), (16, 0.3)],
    "activation": "relu",
    "optimizer": "adam"
}

print(NETWORK_ADDRESSES)

if __name__ == '__main__':

    # Espera inicial para que Docker estabilice la red
    time.sleep(3)

    for round in range(ROUNDS): 
        print(f"\n>>> INICIO RONDA {round} <<<", flush=True)
        
        # 1. COORDINACIÓN
        nodo_id = coordinate(PEERS)
        
        server_ip = NETWORK_ADDRESSES[int(nodo_id) - 1]
        port_ip = int(server_ip.split(':')[1])
        nodo_ip = server_ip.split(':')[0]
        
        print(f"Selected node ID: {nodo_id} address: {nodo_ip}:{port_ip}", flush=True)

        time.sleep(2)  # Pequeña espera antes de iniciar la siguiente fase

        scores = []
        patience = 3
        
        # 2. ENTRENAMIENTO
        if server_ip == DOCKER_ADDRESS:
            # Soy el servidor
            print(f"[MAIN] Iniciando Servidor FL (Esperando {NCLIENTS - 1} clientes)...", flush=True)
            # IMPORTANTE: Usamos BIND_PORT (interno 5000), no DOCKER_PORT (externo)
            server(BIND_PORT, SUB_ROUNDS + 1, NCLIENTS - 1, PARAMS, scores)
        
        else:
            time.sleep(5) 
            print(f"[MAIN] Conectando al servidor {nodo_ip}:{port_ip}...", flush=True)
            client(nodo_ip, port_ip, SUB_ROUNDS + 1)

            converged = checkConvergence(scores, 3)
            if converged:
                print("Convergence reached!!!")
                break

        print(scores)
        with open(f'scores{NODE_ID}.txt', "+a") as f:
            for line in scores:
                f.write(','.join([str(l) for l in line]) + '\n')

        print(f"Round {round} completed!!!", flush=True)
        print("="*60,'\n', flush=True)
    
    print("="*60, '\n')
    print("Federated training completed successfully!!!", flush=True)
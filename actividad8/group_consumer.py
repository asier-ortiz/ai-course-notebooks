import multiprocessing
import subprocess
import os


def run_consumer(instance_id):
    print(f"Lanzando consumidor {instance_id}")
    env = os.environ.copy()
    env["CONSUMER_ID"] = str(instance_id)
    env["PYTHONUNBUFFERED"] = "1"  # Para mostrar salida en tiempo real
    subprocess.run(["python3", "consumer.py"], env=env)


if __name__ == "__main__":
    workers = []
    for i in range(2):  # Simula dos consumidores
        p = multiprocessing.Process(target=run_consumer, args=(i,))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()

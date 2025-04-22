include .env

.PHONY: help build run bash stop restart logs stats clean prune start

CONTAINER_NAME=hdfs-actividad7
IMAGE_NAME=hdfs-actividad7
VOLUME_DIR=$(shell pwd)/data
PLATFORM=linux/amd64

help:
	@echo "Opciones disponibles:"
	@echo "-----------------------------"
	@echo " build       - Construir la imagen Docker"
	@echo " run         - Ejecutar el contenedor en segundo plano"
	@echo " start       - Construir y lanzar Hadoop funcionando automáticamente"
	@echo " bash        - Entrar en bash dentro del contenedor"
	@echo " stop        - Detener el contenedor"
	@echo " restart     - Reiniciar contenedor"
	@echo " logs        - Ver los logs del contenedor"
	@echo " stats       - Ver estadísticas en vivo del contenedor"
	@echo " clean       - Eliminar la imagen y contenedores"
	@echo " prune       - Limpiar redes Docker huérfanas"
	@echo "-----------------------------"

build:
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME) .

run:
	@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	docker run --platform $(PLATFORM) -d \
		-p 50070:50070 \
		-p 9000:9000 \
		-v $(VOLUME_DIR):/data \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME)

bash:
	docker exec -it $(CONTAINER_NAME) bash

stop:
	docker stop $(CONTAINER_NAME)

restart: stop run

logs:
	docker logs -f $(CONTAINER_NAME)

stats:
	docker stats $(CONTAINER_NAME)

clean:
	- docker ps -a -q --filter "ancestor=$(IMAGE_NAME)" | xargs -r docker rm -f
	docker rmi -f $(IMAGE_NAME)

prune:
	docker network prune -f

start: build run
	@echo "Contenedor arrancado. Puedes consultar Hadoop en http://localhost:50070"
include .env

.PHONY: help build run bash stop restart logs stats clean prune

CONTAINER_NAME=hdfs-actividad7
IMAGE_NAME=hdfs-actividad7
VOLUME_DIR=$(shell pwd)/data

help:
	@echo "Opciones disponibles:"
	@echo "-----------------------------"
	@echo " build       - Construir la imagen Docker"
	@echo " run         - Ejecutar el contenedor interactivo"
	@echo " bash        - Entrar en bash dentro del contenedor corriendo"
	@echo " stop        - Detener el contenedor"
	@echo " restart     - Reiniciar contenedor"
	@echo " logs        - Ver los logs del contenedor"
	@echo " stats       - Ver estadísticas en vivo del contenedor"
	@echo " clean       - Eliminar la imagen"
	@echo " prune       - Limpiar redes Docker huérfanas"
	@echo "-----------------------------"

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it --rm \
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
	docker rmi $(IMAGE_NAME)

prune:
	docker network prune -f
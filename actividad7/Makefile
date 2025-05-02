include .env

.PHONY: help build run bash stop restart logs stats clean prune start reset actividad-log

# Ruta absoluta del directorio actual del Makefile
BASE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# Variables de entorno desde .env
CONTAINER_NAME ?= hdfs-actividad7
IMAGE_NAME ?= hdfs-actividad7
PLATFORM ?= linux/amd64
INPUT_DIR ?= /home/ubuntu/bigdata/examples
VOLUME_DIR ?= $(BASE_DIR)/home/ubuntu/bigdata/examples
HDFS_USER_NAME ?= default_user

help:
	@echo "Opciones disponibles:"
	@echo "-----------------------------"
	@echo " build           - Construir la imagen Docker"
	@echo " run             - Ejecutar el contenedor en segundo plano"
	@echo " start           - Construir y lanzar Hadoop funcionando automáticamente"
	@echo " bash            - Entrar en bash dentro del contenedor"
	@echo " stop            - Detener el contenedor"
	@echo " restart         - Reiniciar contenedor"
	@echo " logs            - Ver los logs del contenedor"
	@echo " stats           - Ver estadísticas en vivo del contenedor"
	@echo " clean           - Eliminar la imagen y contenedores"
	@echo " prune           - Limpiar redes Docker huérfanas"
	@echo " reset           - Eliminar contenedor, imagen y archivos de entrada"
	@echo " actividad-log   - Mostrar el log de actividad7.sh desde dentro del contenedor"
	@echo "-----------------------------"

build:
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME) $(BASE_DIR)

run:
	@test -d $(VOLUME_DIR) || (echo "La ruta $(VOLUME_DIR) no existe. Revisa la estructura del proyecto." && exit 1)
	@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	docker run --platform $(PLATFORM) -d \
		-p 50070:50070 \
		-p 9000:9000 \
		-v $(VOLUME_DIR):$(INPUT_DIR) \
		-e HDFS_USER_NAME=$(HDFS_USER_NAME) \
		-e INPUT_DIR=$(INPUT_DIR) \
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

reset:
	@echo "Eliminando contenedor, imagen y archivos de entrada..."
	- docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	- docker rmi -f $(IMAGE_NAME) 2>/dev/null || true
	- rm -rf $(VOLUME_DIR)/*
	@echo "Entorno limpio. Ejecuta 'make start' para reconstruir y reiniciar."

actividad-log:
	@echo "Mostrando el log de actividad7.sh..."
	docker exec -it $(CONTAINER_NAME) bash -c "cat /var/log/actividad7.log || echo 'Log no encontrado o aún no generado.'"
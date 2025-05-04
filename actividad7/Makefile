include .env

.PHONY: help build up down restart bash logs stats clean prune start reset actividad-log

BASE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
COMPOSE := docker compose

help:
	@echo "Opciones disponibles:"
	@echo "-----------------------------"
	@echo " build           - Construir la imagen"
	@echo " up              - Levantar el servicio"
	@echo " start           - Construir y lanzar todo automáticamente"
	@echo " bash            - Entrar en bash dentro del contenedor"
	@echo " logs            - Ver los logs del contenedor"
	@echo " stats           - Ver estadísticas en vivo del contenedor"
	@echo " down            - Detener y eliminar el contenedor"
	@echo " clean           - Eliminar imagen y contenedor"
	@echo " prune           - Limpiar redes Docker huérfanas"
	@echo " reset           - Eliminar todo y limpiar carpeta de ejemplos"
	@echo " actividad-log   - Mostrar el log de actividad7.sh desde el contenedor"
	@echo "-----------------------------"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

start: build up
	@echo "Contenedor arrancado. Puedes consultar Hadoop en:"
	@echo "- NameNode UI: http://localhost:50070"

bash:
	docker exec -it $(CONTAINER_NAME) bash

logs:
	$(COMPOSE) logs -f

stats:
	docker stats $(CONTAINER_NAME)

down:
	$(COMPOSE) down

clean: down
	@docker rmi -f $(IMAGE_NAME) || true

prune:
	docker network prune -f

reset: clean
	@echo "Limpiando archivos del volumen..."
	@rm -rf $(BASE_DIR)/home/ubuntu/bigdata/examples/*
	@echo "Entorno limpio. Ejecuta 'make start' para reconstruir todo."

actividad-log:
	docker exec -it $(CONTAINER_NAME) bash -c "cat /var/log/actividad7.log || echo 'Log no encontrado o aún no generado.'"
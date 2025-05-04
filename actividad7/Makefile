.PHONY: help build up down restart bash logs stats clean prune start reset actividad-log actividad-run

BASE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

help:
	@echo "Opciones disponibles:"
	@echo "-----------------------------"
	@echo " build           - Construir todas las imágenes"
	@echo " up              - Levantar todos los servicios en segundo plano"
	@echo " start           - Construir y lanzar el clúster completo"
	@echo " bash            - Entrar al contenedor del NameNode"
	@echo " logs            - Ver logs de todos los servicios"
	@echo " stats           - Ver estadísticas en tiempo real"
	@echo " down            - Detener y eliminar todos los contenedores"
	@echo " clean           - Eliminar contenedores, volúmenes, imágenes y redes huérfanas"
	@echo " prune           - Limpiar redes Docker huérfanas"
	@echo " reset           - Eliminar todo y limpiar datos de ejemplos"
	@echo " actividad-run   - Ejecutar actividad7.sh una vez"
	@echo " actividad-log   - Mostrar el log de actividad7.sh"
	@echo "-----------------------------"

build:
	docker compose build

up:
	docker compose up -d

start: build up
	@echo "Clúster levantado correctamente."
	@echo "- NameNode UI: http://localhost:9870"
	@echo "- YARN UI: http://localhost:8088"
	@echo "- HistoryServer: http://localhost:19888"

bash:
	docker exec -it namenode bash

logs:
	docker compose logs -f

stats:
	docker stats namenode datanode resourcemanager nodemanager historyserver

down:
	docker compose down

clean:
	docker compose down -v --rmi all --remove-orphans
	@docker rmi -f hdfs-actividad7 || true

prune:
	docker network prune -f

reset: clean
	@echo "Limpiando archivos del volumen..."
	@rm -rf $(BASE_DIR)/home/ubuntu/bigdata/examples/*
	@echo "Entorno limpio. Ejecuta 'make start' para reconstruir todo."

actividad-run:
	docker compose run --rm actividad7

actividad-log:
	docker exec -it namenode bash -c "cat /var/log/actividad7.log || echo 'Log no encontrado o aún no generado.'"
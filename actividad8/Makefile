include .env

.PHONY: help up down restart bash logs stats ps prune clean

help:
	@echo "Opciones disponibles:"
	@echo "-----------------------------"
	@echo " up              - Levantar el entorno Kafka + Zookeeper"
	@echo " down            - Parar y eliminar los contenedores"
	@echo " restart         - Reiniciar todo el entorno"
	@echo " bash-kafka      - Entrar en bash dentro del contenedor de Kafka"
	@echo " bash-zookeeper  - Entrar en bash dentro del contenedor de Zookeeper"
	@echo " logs-kafka      - Ver los logs del contenedor de Kafka"
	@echo " logs-zookeeper  - Ver los logs del contenedor de Zookeeper"
	@echo " logs            - Ver los logs de todos los servicios"
	@echo " stats           - Ver estadísticas de los contenedores"
	@echo " ps              - Ver el estado de los contenedores"
	@echo " prune           - Eliminar redes Docker huérfanas"
	@echo " clean           - Borrar todo: contenedores + imágenes"
	@echo "-----------------------------"

up:
	docker compose up -d

down:
	docker compose down

restart: down up

bash-kafka:
	docker exec -it $(KAFKA_CONTAINER_NAME) bash

bash-zookeeper:
	docker exec -it $(ZOOKEEPER_CONTAINER_NAME) bash

logs-kafka:
	docker logs -f $(KAFKA_CONTAINER_NAME)

logs-zookeeper:
	docker logs -f $(ZOOKEEPER_CONTAINER_NAME)

logs:
	docker compose logs -f

stats:
	docker stats $(ZOOKEEPER_CONTAINER_NAME) $(KAFKA_CONTAINER_NAME)

ps:
	docker ps --filter "name=$(ZOOKEEPER_CONTAINER_NAME)" --filter "name=$(KAFKA_CONTAINER_NAME)"

prune:
	docker network prune -f

clean: down
	docker rmi -f confluentinc/cp-zookeeper:$(ZOOKEEPER_VERSION) confluentinc/cp-kafka:$(KAFKA_VERSION) || true
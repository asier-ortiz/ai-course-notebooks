# Kafka CLI Commands Guide

This guide contains a collection of useful **Kafka command-line operations**, adaptable to local setups or remote brokers.  
Replace `<BROKER_URL>` with your actual broker address (e.g., `localhost:9092`, or `my-broker:9092`).

---

## Topics Management

- **List topics**
  ```bash
  kafka-topics.sh --list --bootstrap-server <BROKER_URL>
  ```

- **Create a topic**
  ```bash
  kafka-topics.sh --create --bootstrap-server <BROKER_URL> --replication-factor 1 --partitions 1 --topic my-topic
  ```

- **Describe a topic**
  ```bash
  kafka-topics.sh --describe --bootstrap-server <BROKER_URL> --topic my-topic
  ```

- **Delete a topic**
  ```bash
  kafka-topics.sh --delete --bootstrap-server <BROKER_URL> --topic my-topic
  ```

---

## Producers and Consumers

- **Start a producer**
  ```bash
  kafka-console-producer.sh --broker-list <BROKER_URL> --topic my-topic
  ```

- **Start a consumer (from beginning)**
  ```bash
  kafka-console-consumer.sh --bootstrap-server <BROKER_URL> --topic my-topic --from-beginning
  ```

- **Start a consumer (only new messages)**
  ```bash
  kafka-console-consumer.sh --bootstrap-server <BROKER_URL> --topic my-topic
  ```

---

## Consumer Groups

- **List all consumer groups**
  ```bash
  kafka-consumer-groups.sh --bootstrap-server <BROKER_URL> --list
  ```

- **Describe a specific consumer group**
  ```bash
  kafka-consumer-groups.sh --bootstrap-server <BROKER_URL> --describe --group my-group
  ```

- **Delete a consumer group**
  ```bash
  kafka-consumer-groups.sh --bootstrap-server <BROKER_URL> --delete --group my-group
  ```

---

## Partitions and Offsets

- **Manually set consumer offsets**
  ```bash
  kafka-consumer-groups.sh --bootstrap-server <BROKER_URL> --group my-group --topic my-topic --reset-offsets --to-earliest --execute
  ```

- **List partition assignments**
  ```bash
  kafka-topics.sh --bootstrap-server <BROKER_URL> --describe --topic my-topic
  ```

---

## Broker and Cluster Info

- **Check broker API versions**
  ```bash
  kafka-broker-api-versions.sh --bootstrap-server <BROKER_URL>
  ```

- **View cluster metadata**
  ```bash
  kafka-metadata-shell.sh --bootstrap-server <BROKER_URL>
  ```

---

## Miscellaneous

- **Send a test message via echo**
  ```bash
  echo "Hello Kafka" | kafka-console-producer.sh --broker-list <BROKER_URL> --topic my-topic
  ```

- **Consume from specific partition**
  ```bash
  kafka-console-consumer.sh --bootstrap-server <BROKER_URL> --topic my-topic --partition 0
  ```

- **Consume with JSON formatting**
  ```bash
  kafka-console-consumer.sh --bootstrap-server <BROKER_URL> --topic my-topic --from-beginning --property print.key=true --property key.separator=" : "
  ```

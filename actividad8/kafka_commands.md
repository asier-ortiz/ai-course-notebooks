
# Kafka Commands Guide

This guide contains a collection of useful **Kafka** command-line operations.

---

## Topics Management

- **List topics**
  ```bash
  kafka-topics.sh --list --bootstrap-server localhost:9092
  ```

- **Create a topic**
  ```bash
  kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic my-topic
  ```

- **Describe a topic**
  ```bash
  kafka-topics.sh --describe --bootstrap-server localhost:9092 --topic my-topic
  ```

- **Delete a topic**
  ```bash
  kafka-topics.sh --delete --bootstrap-server localhost:9092 --topic my-topic
  ```

---

## Producers and Consumers

- **Start a producer**
  ```bash
  kafka-console-producer.sh --broker-list localhost:9092 --topic my-topic
  ```

- **Start a consumer**
  ```bash
  kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic my-topic --from-beginning
  ```

---

## Consumer Groups

- **List consumer groups**
  ```bash
  kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list
  ```

- **Describe a consumer group**
  ```bash
  kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group my-group
  ```

- **Delete a consumer group**
  ```bash
  kafka-consumer-groups.sh --bootstrap-server localhost:9092 --delete --group my-group
  ```

---

## Other Useful Commands

- **Check broker status (inside the container)**
  ```bash
  kafka-broker-api-versions.sh --bootstrap-server localhost:9092
  ```

- **Send messages manually from producer to consumer (testing)**
  ```bash
  echo "Test message" | kafka-console-producer.sh --broker-list localhost:9092 --topic my-topic
  ```

- **Consume only new messages (not from beginning)**
  ```bash
  kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic my-topic
  ```

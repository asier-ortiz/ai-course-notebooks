#!/bin/bash
set -e  # Termina el script si algún comando falla

# Configuro las variables de entorno necesarias para Hadoop
export HADOOP_HOME=/opt/hadoop-2.7.4
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# Función que espera a que HDFS esté disponible antes de continuar
wait_for_hdfs() {
  echo "Esperando a que HDFS esté operativo..."
  until hdfs dfs -ls / >/dev/null 2>&1; do
      echo "HDFS aún no responde, esperando 5s..."
      sleep 5
  done
  echo "HDFS está operativo."
}

# En función del parámetro recibido desde docker-compose, lanzo el servicio correspondiente
case "$1" in

  start-namenode)
    echo "==> Iniciando NameNode"
    # Formateo el NameNode solo si no está ya inicializado
    if [ ! -d "$HADOOP_HOME/data/namenode/current" ]; then
      echo "Formateando NameNode..."
      hdfs namenode -format -nonInteractive
    fi
    exec hdfs namenode
    ;;

  start-datanode)
    echo "==> Iniciando DataNode"
    exec hdfs datanode
    ;;

  start-resourcemanager)
    echo "==> Iniciando ResourceManager"
    wait_for_hdfs
    exec yarn resourcemanager
    ;;

  start-nodemanager)
    echo "==> Iniciando NodeManager"
    wait_for_hdfs
    exec yarn nodemanager
    ;;

  start-historyserver)
    echo "==> Iniciando HistoryServer"
    wait_for_hdfs
    exec mapred historyserver
    ;;

  start-hdfs-actividad7)
    echo "==> Modo actividad7: NameNode + DataNode + carga de datos"

    # Formateo si no existe el NameNode ya inicializado
    if [ ! -d "$HADOOP_HOME/data/namenode/current" ]; then
      echo "Formateando NameNode..."
      hdfs namenode -format -nonInteractive
    fi

    echo "Iniciando NameNode..."
    hdfs namenode &

    echo "Iniciando DataNode..."
    hdfs datanode &

    wait_for_hdfs

    echo "Ejecutando actividad7.sh..."
    if /setup/actividad7.sh > /var/log/actividad7.log 2>&1; then
        echo "actividad7.sh finalizado correctamente." | tee -a /var/log/actividad7.log
    else
        echo "Error al ejecutar actividad7.sh" | tee -a /var/log/actividad7.log
    fi

    wait -n  # Espero a que uno de los procesos finalice
    ;;

  *)
    echo "Comando no reconocido: $1"
    echo "Opciones válidas: start-namenode | start-datanode | start-resourcemanager | start-nodemanager | start-historyserver | start-hdfs-actividad7"
    exec "$@"
    ;;
esac
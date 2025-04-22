#!/bin/bash
set -e

# Variables de entorno
export HADOOP_HOME=/opt/hadoop-2.7.4
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# Formatear el NameNode si no está formateado
if [ ! -d "$HADOOP_HOME/data/namenode/current" ]; then
  echo "Formateando NameNode..."
  hdfs namenode -format -nonInteractive
fi

# Lanzar NameNode y DataNode en segundo plano
echo "Iniciando NameNode..."
hdfs namenode &

echo "Iniciando DataNode..."
hdfs datanode &

# Mantener el contenedor activo mientras cualquier proceso siga vivo
wait -n
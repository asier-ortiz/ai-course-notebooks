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

# Iniciar NameNode y DataNode en segundo plano
echo "Iniciando NameNode..."
hdfs namenode &

echo "Iniciando DataNode..."
hdfs datanode &

# Esperar activamente a que HDFS esté listo
echo "Esperando a que HDFS esté operativo..."
until hdfs dfs -ls / >/dev/null 2>&1; do
    echo "HDFS aún no responde, esperando 5s..."
    sleep 5
done
echo "HDFS está operativo."

# Ejecutar la actividad automatizada y guardar la salida en un log para depuración.
# Si el script falla, se mostrará un aviso, pero el contenedor seguirá funcionando.
echo "Ejecutando actividad7.sh..."
if /setup/actividad7.sh > /var/log/actividad7.log 2>&1; then
    echo "actividad7.sh finalizado correctamente." | tee -a /var/log/actividad7.log
else
    echo "Error al ejecutar actividad7.sh" | tee -a /var/log/actividad7.log
fi

# Mantener el contenedor activo
wait -n
FROM ubuntu:20.04

# Evitar preguntas en la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalación de Java, SSH, y utilidades necesarias
RUN apt-get update && apt-get install -y \
    openjdk-8-jdk \
    openssh-server \
    wget \
    curl \
    unzip \
    vim \
    net-tools \
    procps \
    iputils-ping \
    && apt-get clean

# Variables de entorno
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV HADOOP_VERSION=2.7.4
ENV HADOOP_HOME=/opt/hadoop-2.7.4

# Descargar e instalar Hadoop
RUN wget https://archive.apache.org/dist/hadoop/core/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -P /tmp && \
    tar -xzf /tmp/hadoop-${HADOOP_VERSION}.tar.gz -C /opt && \
    rm /tmp/hadoop-${HADOOP_VERSION}.tar.gz

# Crear carpetas necesarias
RUN mkdir -p /opt/hadoop-2.7.4/data/namenode /opt/hadoop-2.7.4/data/datanode

# Crear estructura de archivos esperada (para evitar errores si no se monta el volumen)
RUN mkdir -p /home/ubuntu/bigdata/examples

# Copiar configuración
COPY hadoop-config/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
COPY hadoop-config/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml

# Copiar scripts
COPY entrypoint.sh /entrypoint.sh
COPY setup/actividad7.sh /setup/actividad7.sh

# Permisos
RUN chmod +x /entrypoint.sh /setup/actividad7.sh

# Añadir variables al bashrc para acceso interactivo
RUN echo 'export HADOOP_HOME=/opt/hadoop-2.7.4' >> /root/.bashrc && \
    echo 'export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop' >> /root/.bashrc && \
    echo 'export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin' >> /root/.bashrc

# Puerto para acceder al NameNode WebUI
EXPOSE 50070

# Comando por defecto
CMD ["/entrypoint.sh"]
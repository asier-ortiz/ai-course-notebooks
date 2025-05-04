# Imagen base: Ubuntu 20.04 compatible con Hadoop 2.7.4 y Java 8
FROM ubuntu:20.04

# Evitar prompts interactivos en la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalo Java 8, SSH y herramientas básicas para red, depuración y sistema
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
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Añado enlace simbólico para que JAVA_HOME esté donde Hadoop lo espera
RUN ln -s /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/default-java || true

# Variables de entorno necesarias para Java y Hadoop
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$JAVA_HOME/bin:$PATH
ENV HADOOP_VERSION=2.7.4
ENV HADOOP_HOME=/opt/hadoop-${HADOOP_VERSION}

# Descargo y descomprimo Hadoop en /opt
RUN wget https://archive.apache.org/dist/hadoop/core/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -P /tmp && \
    tar -xzf /tmp/hadoop-${HADOOP_VERSION}.tar.gz -C /opt && \
    rm /tmp/hadoop-${HADOOP_VERSION}.tar.gz

# Creo directorios para NameNode, DataNode, timeline server y datos del proyecto
RUN mkdir -p \
    $HADOOP_HOME/data/namenode \
    $HADOOP_HOME/data/datanode \
    /hadoop/yarn/timeline \
    /home/ubuntu/bigdata/examples \
    /var/log/hadoop

# Copio los ficheros de configuración personalizados de Hadoop
COPY hadoop-config/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
COPY hadoop-config/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml
COPY hadoop-config/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml
COPY hadoop-config/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml

# Copio los scripts de arranque y el de la actividad
COPY entrypoint.sh /entrypoint.sh
COPY setup/actividad7.sh /setup/actividad7.sh

# Doy permisos de ejecución a los scripts
RUN chmod +x /entrypoint.sh /setup/actividad7.sh

# Añado variables de entorno al bashrc por si necesito entrar en el contenedor
RUN echo 'export HADOOP_HOME=/opt/hadoop-2.7.4' >> /root/.bashrc && \
    echo 'export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop' >> /root/.bashrc && \
    echo 'export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$JAVA_HOME/bin' >> /root/.bashrc

# Exponer puertos para NameNode, DataNode, ResourceManager, NodeManager y HistoryServer
EXPOSE 50070 9870 9000 8088 8042 19888

# Punto de entrada común para lanzar los servicios definidos
ENTRYPOINT ["/entrypoint.sh"]
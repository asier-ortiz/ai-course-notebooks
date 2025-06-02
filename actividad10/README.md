# Actividad 10 – Spark y análisis de datos

Este ejercicio utiliza PySpark, que requiere Java 11 para funcionar correctamente. Si tienes otra versión de Java por defecto (por ejemplo, Java 17 o 21), es posible que obtengas errores al iniciar una sesión Spark. Para evitar conflictos, se incluye un script que configura temporalmente Java 11 solo mientras se ejecuta el notebook.

## Requisitos previos

- Python 3.9 o superior
- Java 11 instalado en el sistema (además de tu versión actual)
- Entorno virtual configurado (`.venv`)
- Paquetes instalados: `pyspark`, `jupyter`, etc.

## Cómo ejecutar la actividad

1. Abre una terminal y accede a la carpeta de la actividad:

```bash
cd ai-course-notebooks/actividad10
```

2. Dale permisos de ejecución al script si aún no los tiene:

```bash
chmod +x run_spark.sh
```

3. Ejecuta el script:

```bash
./run_spark.sh
```

Este script hace lo siguiente:

- Establece temporalmente Java 11 como entorno activo (sin afectar tu sistema).
- Muestra la versión de Java utilizada.
- Activa el entorno virtual Python.
- Lanza Jupyter Notebook en el navegador.

## Verificación

Cuando se ejecute correctamente, deberías ver en la terminal algo como:

```
Using Java:
java version "11.x.x"
...
```

Y el entorno de Jupyter se abrirá automáticamente.

## Errores comunes

- **Otro SparkContext activo**: Si aparece una advertencia como `Another SparkContext is being constructed...`, asegúrate de no tener otra instancia previa activa. Reinicia el kernel del notebook.

- **`NoClassDefFoundError` o errores con `ByteArrayMethods`**: Suele indicar que Java no está correctamente configurado. Asegúrate de usar Java 11 mediante el script.

- **`JAVA_HOME` vacío**: Si ejecutas directamente Jupyter sin el script, es posible que no esté definida la variable JAVA_HOME con la versión adecuada. Usa siempre `run_spark.sh`.


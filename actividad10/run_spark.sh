#!/bin/bash
export JAVA_HOME=$(/usr/libexec/java_home -v11)
export PATH="$JAVA_HOME/bin:$PATH"
echo "Using Java:"
java -version
source .venv/bin/activate
jupyter notebook
#!/bin/sh
# Gradle start up script for UN*X
set -e

APP_BASE_NAME=$(basename "$0")
APP_HOME=$(cd "$(dirname "$0")" && pwd -P)
CLASSPATH="$APP_HOME/gradle/wrapper/gradle-wrapper.jar"

# Find java
if [ -n "$JAVA_HOME" ]; then
  JAVACMD="$JAVA_HOME/bin/java"
else
  JAVACMD="java"
fi

if ! command -v "$JAVACMD" > /dev/null 2>&1; then
  echo "ERROR: JAVA_HOME is not set and 'java' was not found in PATH." >&2
  exit 1
fi

# Build JVM opts — no embedded quotes, plain flags only
JVM_OPTS="-Xmx64m -Xms64m"

# Allow override via GRADLE_OPTS or JAVA_OPTS
exec "$JAVACMD" \
  $JVM_OPTS \
  ${JAVA_OPTS:-} \
  ${GRADLE_OPTS:-} \
  "-Dorg.gradle.appname=$APP_BASE_NAME" \
  -classpath "$CLASSPATH" \
  org.gradle.wrapper.GradleWrapperMain \
  "$@"

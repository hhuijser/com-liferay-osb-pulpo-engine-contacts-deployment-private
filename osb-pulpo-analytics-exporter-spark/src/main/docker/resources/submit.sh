#!/bin/bash

cd ${SPARK_USER_HOME}

if test -z "$SPARK_MASTER_URL"
then
	echo "SPARK_MASTER_URL can not be empty"
	exit 1
fi

if test -z "$APPLICATION_ID"
then
	echo "APPLICATION_ID can not be empty"
	exit 1
fi

if test -z "$CASSANDRA_CONNECTION_HOST"
then
	echo "CASSANDRA_CONNECTION_HOST can not be empty"
	exit 1
fi

CONTAINER_IP="$(ifconfig | grep -Eo 'inet (addr:)?([0-9]+\.){3}[0-9]+' | grep -Eo '([0-9]+\.){3}[0-9]+' | grep -v '127.0.0.1' | head)"

if test -z "$CONTAINER_IP"
then
  echo "Cannot determine the container IP address."
  exit 1
fi

JOB_JAR=`find ${SPARK_USER_HOME} -maxdepth 1 -name "*.jar"`

SPARK_CONF="--master ${SPARK_MASTER_URL} --conf spark.driver.host=$SPARK_DRIVER_HOST --conf spark.driver.port=$SPARK_DRIVER_PORT"
SPARK_CONF="${SPARK_CONF} --conf spark.ui.port=$SPARK_UI_PORT --conf spark.driver.blockManager.port=$SPARK_DRIVER_BLOCKMGR_PORT"
SPARK_CONF="${SPARK_CONF} --conf spark.driver.bindAddress=$CONTAINER_IP ${SPARK_CONF_CUSTOM}"

SPARK_DRIVER_OPT="-DapplicationId=${APPLICATION_ID}"
SPARK_DRIVER_OPT="${SPARK_DRIVER_OPT} -Duser.timezone=GMT -Duser.language=en -Duser.country=GB"
SPARK_DRIVER_OPT="${SPARK_DRIVER_OPT} -Dlcs.spark.cassandra.connection.host=${CASSANDRA_CONNECTION_HOST}"
SPARK_DRIVER_OPT="${SPARK_DRIVER_OPT} ${SPARK_DRIVER_OPT_CUSTOM}"

SPARK_EXEC="${SPARK_HOME}/bin/spark-submit"

SPARK_PACKAGES="datastax:spark-cassandra-connector:2.0.0-s_2.11"

if test -n "$SPARK_PACKAGES_CUSTOM"
then
	SPARK_PACKAGES="${SPARK_PACKAGES},$SPARK_PACKAGES_CUSTOM"
fi

echo "Submit date: $NOW"
echo "Application Id: ${APPLICATION_ID}"
echo "Spark Conf: ${SPARK_CONF} ${SPARK_DRIVER_OPT}"
echo "Spark Application Jar: ${JOB_JAR}"

echo "Executing: $SPARK_EXEC $SPARK_CONF --driver-java-options \"$SPARK_DRIVER_OPT\" --packages $SPARK_PACKAGES $JOB_JAR"

$SPARK_EXEC $SPARK_CONF --driver-java-options "$SPARK_DRIVER_OPT" --packages $SPARK_PACKAGES $JOB_JAR

cd -
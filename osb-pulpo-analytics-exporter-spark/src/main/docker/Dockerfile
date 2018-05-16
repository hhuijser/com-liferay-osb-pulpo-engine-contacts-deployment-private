FROM liferay/com-liferay-osb-lcs-spark-base:2.2.0
MAINTAINER Riccardo Ferrari riccardo.ferrari@liferay.com

USER root

ADD *.jar /opt/spark/
ADD resources/submit.sh /opt/spark/submit.sh

RUN rm -Rf ${SPARK_USER_HOME}/lib
RUN chown -R spark:spark *.jar submit.sh
RUN chmod 0755 submit.sh

USER spark

#RUN cp ${SPARK_HOME}/conf/log4j.properties.template ${SPARK_HOME}/conf/log4j.properties
#RUN echo "log4j.logger.org.springframework=DEBUG" >> ${SPARK_HOME}/conf/log4j.properties

ENTRYPOINT ${SPARK_USER_HOME}/submit.sh
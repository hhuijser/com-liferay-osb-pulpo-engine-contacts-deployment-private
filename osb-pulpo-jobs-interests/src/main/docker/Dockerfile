FROM python:3.5.5-slim

RUN mkdir jupyter

ADD jupyter /jupyter/
ADD resources/submit.sh /

USER root

RUN apt-get update && apt-get install -y python-pip && \
pip3 install nbconvert && \
pip3 install jupyter && \
pip3 install spacy  && python -m spacy download en && \
pip3 install pandas && \
pip3 install langdetect && \
pip3 install furl && \
pip3 install boto3 && \
pip3 install smart_open

ENTRYPOINT /submit.sh
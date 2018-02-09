# Liferay OSB Pulpo Jobs

## Interests Docker Image

1. Set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables with the AWS credentials.
2. Go to `osb-pulpo-jobs-interests`.
3. Run `../gradlew createDocker` and wait for the Docker image to be built.
4. Run `../gradlew startDocker && docker logs -f com-liferay-osb-pulpo-jobs-interests-private` to run the image and show the container logs.

To run the interests script locally, run:

```
apt-get update && apt-get install -y python-pip
pip3 install nbconvert
pip3 install jupyter
pip3 install spacy  && python -m spacy download en
pip3 install pandas
pip3 install langdetect
pip3 install furl
pip3 install boto3
pip3 install smart_open

jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 jupyter/interests.ipynb
```
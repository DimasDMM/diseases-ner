FROM python:3.7.5-buster

# Workdir
ENV WORKDIR /App
WORKDIR $WORKDIR

# Install/update dependencies
RUN apt-get update && \
    apt-get install -f -y postgresql-client && \
    apt-get -y install build-essential
RUN python -m pip install --upgrade pip

# Install Python libraries
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git
RUN python -m spacy download en_core_web_sm

# Do not stop the container
ENTRYPOINT ["tail", "-f", "/dev/null"]

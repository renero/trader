# Base image Debian stretch python 3.6
FROM python:3.7.4-slim-buster

# Update Ubuntu Software repository
RUN apt-get update
RUN apt-get install -y vim

# Upgrade pip
RUN pip install --upgrade pip

# Create folder structure and install requirements
RUN mkdir -p /trader
RUN mkdir -p /trader/output

ADD ./requirements.txt /trader
WORKDIR /trader
RUN pip3 install -r requirements.txt

# Add code and update the working directory
ADD . /trader
WORKDIR /trader
VOLUME /trader

# Run the application with unbuffered output to see it on real time
ENV PYTHONPATH "${PYTHONPATH}:/trader:/trader/utils:/trader/predictor:/trader/trader:/trader/indicators:/trader/retriever:/trader/updater"
CMD /trader/pipeline.sh

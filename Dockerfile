# Base image Debian stretch python 3.6
FROM python:3.7.4-slim-buster

# Update Ubuntu Software repository
RUN apt-get update
RUN apt-get install -y vim

# Upgrade pip
RUN pip install --upgrade pip

# Create folder structure and install requirements
RUN mkdir -p /trader
# RUN mkdir -p /trader/output

ADD ./requirements.txt /trader
WORKDIR /trader
RUN pip3 install -r requirements.txt

# Add code and update the working directory
ADD . /trader
WORKDIR /trader
VOLUME /trader
VOLUME /Users/renero/trader/data:/trader/data
VOLUME /Users/renero/trader/output:/trader/output
VOLUME /Users/renero/trader/stasging:/trader/staging

# Run the application with unbuffered output to see it on real time
ENV PYTHONPATH "${PYTHONPATH}:/trader:/trader/utils:/trader/predictor:/trader/trader:/trader/indicators:/trader/retriever:/trader/updater"
CMD /trader/pipeline.sh

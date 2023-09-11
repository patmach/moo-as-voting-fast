FROM python:3.9.10-slim-buster

RUN apt-get update 
RUN apt-get install --reinstall build-essential -y
RUN apt-get install curl -y
RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install msodbcsql18 unixodbc-dev -y
RUN apt-get update 
RUN apt-get install libfreetype6-dev -y
RUN apt-get install pkg-config -y
RUN apt-get install libpng-dev -y
RUN apt-get install pkg-config -y


RUN pip install --upgrade pip
COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN pip install gunicorn
RUN pip install protobuf==3.20.*

#RUN ls - l /opt/mssql/lib/ && sleep 20
#RUN cat libsqlvdi.so && sleep 20
COPY . /app
WORKDIR /app
ENTRYPOINT "./gunicorn.sh"
FROM python:3.10

RUN apt install -y git

WORKDIR /usr/src/app
RUN pip3 install -U pip setuptools

COPY . .
RUN pip3 install -e .[testing]

RUN py.test .

ENTRYPOINT [ "bmws" ]

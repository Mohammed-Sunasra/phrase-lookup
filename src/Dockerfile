FROM python:3.6-slim
RUN apt-get update && apt-get install -qq -y \
    build-essential libpq-dev --no-install-recommends

RUN mkdir /src
WORKDIR /src

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
LABEL maintainer="Mohammed Sunasra <mohammed.sunasra@pixeldust.com>" \
      version="1.0"

CMD python predict.py




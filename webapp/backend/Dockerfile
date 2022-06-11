FROM ubuntu

RUN apt-get -y update
RUN apt-get -y install python3 python3-pip
WORKDIR /app
COPY . .

RUN apt-get install -y wget unzip && \
    wget -P /app https://simon-storage.oss-cn-shanghai.aliyuncs.com/models.zip \
    && unzip -d /app /app/models.zip \
    && rm -f /app/models.zip

RUN pip3 install -r requirements.txt \
    -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

EXPOSE 8000

CMD python3 backend.py
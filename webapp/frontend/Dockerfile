FROM ubuntu

RUN apt-get -y update
RUN apt-get -y install python3 python3-pip

WORKDIR /app

RUN pip3 install -r requirements.txt \
    -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
COPY . .

EXPOSE 8501
CMD streamlit run frontend.py

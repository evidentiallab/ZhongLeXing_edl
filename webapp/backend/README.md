# Backend

## Installation

### download models

```
$wget -P ./ https://simon-storage.oss-cn-shanghai.aliyuncs.com/models.zip
$unzip -d ./ ./models.zip
$rm -rf ./models.zip
```

separate environment such as conda or venv is recommended   

```
$pip install -r requirements.txt
$python backend.py
```

### installation using docker

```
$docker build -t backend .
$docker run -p 8000:8000 backend
```

## Use

visit localhost:8000/docs to test the api 


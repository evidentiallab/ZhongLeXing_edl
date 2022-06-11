# frontend

## Setup

separated environment is recommended.

```
$pip install -r requirements.txt
$streamlit run frontend.py
```

### installation using docker

```
$docker build -t frontend .
$docker run -p 8501:8501 frontend
```

## Use

visit localhost:8501 to access the web application built by streamlit.

## Note

setup.sh is used in Heroku deployment. Visit www.heroku.com to get more information.
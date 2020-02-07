FROM python:3.7-slim-stretch

COPY . /src

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

RUN python /src/app.py

EXPOSE 5000

CMD ["python", "/src/app.py", "serve"]

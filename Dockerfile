FROM python:3.10.7

WORKDIR /app
COPY setup.txt .
COPY . .
RUN pip install --upgrade pip
RUN pip install -r setup.txt


CMD ["python", "main.py"]
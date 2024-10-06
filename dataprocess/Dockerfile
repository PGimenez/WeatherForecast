FROM python:3.10-slim

LABEL org.opencontainers.image.source https://github.com/PGimenez/WeatherForecast
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY process_data.py .

CMD ["python", "process_data.py"]

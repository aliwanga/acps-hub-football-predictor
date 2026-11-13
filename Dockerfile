FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# guard against Windows (CRLF) line endings breaking the shell script
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh
RUN useradd -m app && mkdir -p /app/data /app/model && chown -R app /app
USER app

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/healthz',timeout=4).status==200 else 1)"

# entrypoint.sh trains a first model if none exists, then starts gunicorn
CMD ["./entrypoint.sh"]

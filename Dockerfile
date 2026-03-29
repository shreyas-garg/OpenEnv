FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R user:user /app
USER user

ENV PYTHONPATH=/app
ENV HOME=/home/user

EXPOSE 7860

CMD ["uvicorn", "email_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]

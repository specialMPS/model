FROM python:3.8.5

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirement.txt

EXPOSE 8080

CMD python ./app.py
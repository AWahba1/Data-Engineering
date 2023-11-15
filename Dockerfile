FROM python:3.9

RUN pip install pandas numpy geopy scikit-learn sqlalchemy psycopg2 

WORKDIR /app

CMD ["python","src/main.py"]

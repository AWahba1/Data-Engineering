import pandas as pd
from sqlalchemy import create_engine

def upload_csv_to_postgres(path, table_name, db_username = 'root', db_password = 'root', db_host = 'pgdatabase', db_port = 5432, db_name = 'green_taxis'):
    cleaned_df = pd.read_csv(path)

    engine = create_engine(f'postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')
    if engine.connect():
        print('Connected successfully')
    else:
        print('Failed to connect')

    cleaned_df.to_sql(name=table_name, con=engine, if_exists='fail')

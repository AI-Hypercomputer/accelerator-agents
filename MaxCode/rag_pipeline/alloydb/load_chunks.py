import os
import sys
import json
import argparse
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from google.cloud import storage

def load_parquet_to_alloydb(input_gcs_path, db_config):
    """Reads Parquet files from GCS and inserts into AlloyDB snippets table."""
    
    print(f"Loading data from {input_gcs_path}")
    
    # Load Parquet into Pandas
    # Spark outputs often create a directory with multiple shards
    df = pd.read_parquet(input_gcs_path)
    
    # Pre-process columns to match DB schema
    df['file_extension'] = df['file_path'].apply(lambda x: x.split('.')[-1] if '.' in x else '')
    
    # Convert dict metadata to JSON strings for Postgres JSONB
    df['metadata'] = df['metadata'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else '{}')

    # Connection to AlloyDB
    conn = psycopg2.connect(
        host=db_config['host'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password'],
        port=db_config['port'],
        sslmode='require'
    )
    
    cur = conn.cursor()
    
    # Create table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS chunked_code_snippets (
        id SERIAL PRIMARY KEY,
        repository TEXT,
        branch TEXT,
        commit_hash TEXT,
        file_path TEXT,
        file_extension TEXT,
        code_chunk TEXT,
        chunk_type TEXT,
        metadata JSONB
    );
    """
    print("Ensuring table chunked_code_snippets exists...")
    cur.execute(create_table_query)
    conn.commit() # Commit table creation
    
    # Prepare insertion SQL
    columns = ['repository', 'branch', 'commit_hash', 'file_path', 'file_extension', 'code_chunk', 'chunk_type', 'metadata']
    query = f"INSERT INTO chunked_code_snippets ({', '.join(columns)}) VALUES %s"
    
    # Convert DF to list of tuples for execute_values, ensuring standard Python types
    data_tuples = []
    for row in df[columns].to_dict('records'):
        # Convert metadata dict to JSON string if it is a dict
        if isinstance(row.get('metadata'), dict):
            row['metadata'] = json.dumps(row['metadata'])
            
        # Convert any numpy types to python types
        new_row = []
        for col in columns:
            val = row[col]
            if val is not None and 'numpy' in type(val).__module__:
                if hasattr(val, 'tolist'):
                    val = val.tolist()
                elif hasattr(val, 'item'):
                    val = val.item()
            new_row.append(val)
        data_tuples.append(tuple(new_row))
    
    try:
        print(f"Executing bulk insert for {len(data_tuples)} chunks...")
        execute_values(cur, query, data_tuples)
        conn.commit()
        print("Load successful!")
    except Exception as e:
        conn.rollback()
        print(f"Error during load: {e}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="GCS path to Parquet shard directory")
    parser.add_argument("--db_host", required=True)
    parser.add_argument("--db_name", required=True)
    parser.add_argument("--db_user", required=True)
    parser.add_argument("--db_pass", required=True)
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_pass,
        'port': 5432
    }
    
    load_parquet_to_alloydb(args.input_path, db_config)

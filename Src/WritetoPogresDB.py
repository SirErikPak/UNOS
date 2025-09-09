import pandas as pd
from sqlalchemy import create_engine, text
from io import StringIO
import os

# -----------------------------
# Database Connection Parameters
# -----------------------------
connection_params = {
    'user': 'sir',
    'password': 'password',
    'host': 'localhost',
    'port': 5432,
    'database': 'UNOS'
}

# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql+psycopg2://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}"
)

# ----------------------------------------
# Helper: Load HTML Column Names and DAT Data
# ----------------------------------------
def load_data_with_dynamic_columns(html_path, dat_path):
    """
    Read column names from HTML file and load data from corresponding .DAT file
    """
    with open(html_path, 'r', encoding='latin') as f:
        html_content = f.read()

    if '<table' not in html_content.lower():
        raise ValueError(f"No <table> found in HTML file: {html_path}")

    columns_name_df = pd.read_html(StringIO(html_content))[0]

    if 'LABEL' not in columns_name_df.columns:
        raise ValueError(f"'LABEL' column not found in {html_path}. Available columns: {columns_name_df.columns.tolist()}")

    column_names = list(columns_name_df['LABEL'])

    df = pd.read_csv(
        dat_path,
        sep='\t',
        header=None,
        names=column_names,
        encoding='latin',
        low_memory=False
    )

    return df

# ----------------------------------------
# Dynamic Table Creation
# ----------------------------------------
def create_table_if_not_exists(df, table_name, engine):
    """
    Dynamically create a PostgreSQL table based on DataFrame schema
    """
    dtype_mapping = {
        "object": "TEXT",
        "int64": "BIGINT",
        "float64": "DOUBLE PRECISION",
        "datetime64[ns]": "TIMESTAMP"
    }

    columns_with_types = []
    for col_name, dtype in df.dtypes.items():
        pg_type = dtype_mapping.get(str(dtype), "TEXT")
        safe_col_name = col_name.replace(" ", "_").replace("-", "_")
        columns_with_types.append(f'"{safe_col_name}" {pg_type}')

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        {", ".join(columns_with_types)}
    );
    """

    with engine.begin() as conn:
        conn.execute(text(create_table_sql))

    print(f"Table '{table_name}' created or already exists.")

# ----------------------------------------
# Bulk Insert Using COPY (with proper quoting)
# ----------------------------------------
def bulk_insert(df, table_name, engine):
    """
    Efficient bulk insert into PostgreSQL using COPY with quoted table name
    """
    df.columns = [col.replace(" ", "_").replace("-", "_") for col in df.columns]

    # In-memory CSV
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, header=True)
    csv_buffer.seek(0)

    conn = engine.raw_connection()
    try:
        cursor = conn.cursor()
        # Quote table name to preserve case
        cursor.copy_expert(f'COPY "{table_name}" FROM STDIN WITH CSV HEADER', csv_buffer)
        conn.commit()
    finally:
        conn.close()

    print(f"Data inserted into '{table_name}' successfully.")

# ----------------------------------------
# Main Workflow
# ----------------------------------------
try:
    # Load data from HTML and DAT files
    df_main = load_data_with_dynamic_columns('Data/THORACIC_DATA.htm', 'Data/THORACIC_DATA.DAT')
    df_followup = load_data_with_dynamic_columns('Data/THORACIC_FOLLOWUP_DATA.htm', 'Data/THORACIC_FOLLOWUP_DATA.DAT')
    df_format = load_data_with_dynamic_columns('Data/THORACIC_FORMATS_FLATFILE.htm', 'Data/THORACIC_FORMATS_FLATFILE.DAT')

    # Map datasets to table names
    datasets = {
        "Thoracic_main": df_main,
        "Thoracic_followup": df_followup,
        "Thoracic_format": df_format
    }

    # Process each dataset
    for table_name, df in datasets.items():
        print(f"\nProcessing table: {table_name}")
        create_table_if_not_exists(df, table_name, engine)
        bulk_insert(df, table_name, engine)

    print("\nAll tables processed and data loaded successfully!")

except Exception as e:
    print(f"Error: {e}")

finally:
    engine.dispose()
    print("Database connection closed.")
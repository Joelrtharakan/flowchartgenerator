import psycopg2

conn = psycopg2.connect(
    host="flowchart-db-johnj.postgres.database.azure.com",
    user="flowadmin",
    password="Secure_Route#2026",
    dbname="postgres",      # This is the default DB name in Azure        sslmode="require"       # This is required for Azure connections
)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS flowcharts (
    id SERIAL PRIMARY KEY,
    prompt TEXT,
    mermaid TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
cur.close()
conn.close()

print("Table created successfully")
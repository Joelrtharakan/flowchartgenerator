import json
import psycopg2

# 🔴 UPDATE THESE
DB_HOST = "flowchart-db-johnj.postgres.database.azure.com"
DB_USER = "flowadmin"
DB_PASSWORD = "Secure_Route#2026"
DB_NAME = "postgres"

# connect
conn = psycopg2.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    dbname=DB_NAME
)

cur = conn.cursor()

# file path
FILE_PATH = "/home/jusvin-britto/fg/flowchart_generator/data/data_train.jsonl"

count = 0

with open(FILE_PATH, "r") as f:
    for line in f:
        try:
            item = json.loads(line)

            prompt = item["messages"][1]["content"]
            mermaid = item["messages"][2]["content"]

            cur.execute(
                "INSERT INTO flowcharts (prompt, mermaid) VALUES (%s, %s)",
                (prompt, mermaid)
            )

            count += 1

            # commit every 50 rows (faster)
            if count % 50 == 0:
                conn.commit()
                print(f"{count} rows inserted...")

        except Exception as e:
            print("Error:", e)

# final commit
conn.commit()

cur.close()
conn.close()

print(f"✅ Done! Total inserted: {count}")
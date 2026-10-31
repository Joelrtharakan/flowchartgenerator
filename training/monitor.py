import time
import psycopg2
import subprocess
import os

CHECKPOINTS = [1500, 2000]
triggered = set()

def get_count():
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        dbname="postgres",
        port=5432
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM flowcharts")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count

def trigger_training(count):
    print(f"\n🚀 Training triggered at {count}\n")

    subprocess.run(["python3", "train_phi3.py"])

    # get latest version
    versions = [d for d in os.listdir("models") if d.startswith("v")]
    latest = sorted(versions, key=lambda x: int(x[1:]))[-1]

    # update active model
    with open("current_model.txt", "w") as f:
        f.write(f"models/{latest}")

    print(f"✅ Switched to {latest}")

    # restart API
    subprocess.run(["pkill", "-f", "uvicorn"])

    subprocess.Popen([
        "nohup", "uvicorn", "app:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

while True:
    try:
        count = get_count()
        print("Current:", count)

        for cp in CHECKPOINTS:
            if count >= cp and cp not in triggered:
                trigger_training(count)
                triggered.add(cp)

    except Exception as e:
        print("Error:", e)

    time.sleep(60)

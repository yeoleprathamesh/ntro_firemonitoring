import os
import sys

# Ensure backend directory is in python path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Change current working directory to backend so relative paths work seamlessly
os.chdir(backend_dir)

import sync_daily

if __name__ == "__main__":
    print("=== NTRO Sentinel Daily FIRMS Sync Task Triggered (Root Wrapper) ===")
    fresh_data = sync_daily.fetch_live_firms_data()
    count = sync_daily.process_and_classify(fresh_data)
    print(f"=== Execution Complete: {count} new anomalies integrated into classified_fires.csv ===")

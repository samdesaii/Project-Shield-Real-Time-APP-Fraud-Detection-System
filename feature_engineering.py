import duckdb
import pandas as pd

# 1. Connect to DuckDB (in-memory)
# We don't need a server. This creates a temporary DB in RAM.
con = duckdb.connect(database=':memory:')

# 2. Load Data directly from CSV
# This is the "Magic". We treat the CSV file like a SQL Table.
print("Loading data into DuckDB...")
con.execute("""
    CREATE TABLE transactions AS 
    SELECT * FROM read_csv_auto('transactions.csv');
""")

# 3. The "Risk Engine" Query
# We use Window Functions (OVER PARTITION BY) to look at "past behavior".
query = """
SELECT 
    t.*,
    
    -- FEATURE 1: VELOCITY (Count of txns in last 1 hour)
    -- "Look at this user, order by time, count rows in the last 3600 seconds"
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
    ) - 1 AS count_last_1h,  -- Subtract 1 to exclude 'current' txn
    
    -- FEATURE 2: VELOCITY (Count of txns in last 24 hours)
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW
    ) - 1 AS count_last_24h,
    
    -- FEATURE 3: AVERAGE SPEND (Last 30 days)
    -- Helps us detect if a $5000 txn is weird for THIS user.
    AVG(amount) OVER (
        PARTITION BY customer_id 
        ORDER BY timestamp 
        RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND INTERVAL 1 SECOND PRECEDING
    ) AS avg_amt_last_30d,
    
    -- FEATURE 4: RATIO TO AVERAGE
    -- If avg is $100 and this is $1000, ratio is 10.0 (High Risk)
    CASE 
        WHEN AVG(amount) OVER (
            PARTITION BY customer_id 
            ORDER BY timestamp 
            RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND INTERVAL 1 SECOND PRECEDING
        ) = 0 THEN 0
        ELSE 
            amount / AVG(amount) OVER (
                PARTITION BY customer_id 
                ORDER BY timestamp 
                RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND INTERVAL 1 SECOND PRECEDING
            )
    END AS ratio_to_avg_amt

FROM transactions t
ORDER BY timestamp;
"""

print("Running Feature Engineering SQL...")
df_enriched = con.execute(query).df()

# 4. Cleanup & Save
# Fill NA values (first transaction for a user has no 'previous average')
df_enriched = df_enriched.fillna(0)

print(f"Feature Engineering Complete. Shape: {df_enriched.shape}")
print(df_enriched[['customer_id', 'amount', 'count_last_1h', 'ratio_to_avg_amt']].head())

# Save for the Modeling Phase
df_enriched.to_csv("transactions_enriched.csv", index=False)
print("Saved to transactions_enriched.csv")
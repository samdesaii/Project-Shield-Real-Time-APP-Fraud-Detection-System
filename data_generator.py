import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)

# --- CONFIGURATION ---
NUM_CUSTOMERS = 1000
NUM_TRANSACTIONS = 5000 
FRAUD_RATIO = 0.05

# --- HELPER LISTS ---
BANK_TYPES = ['Traditional', 'Traditional', 'Traditional', 'NeoBank', 'NeoBank', 'Crypto_Exchange']

def generate_customers(n):
    """
    Generates customer base. 
    """
    customers = []
    for _ in range(n):
        customer = {
            "customer_id": fake.uuid4(),
            "name": fake.name(),
            "age": np.random.randint(18, 90),
            "account_balance": round(np.random.uniform(1000, 50000), 2),
            "device_id": fake.uuid4(),
            "account_age_days": np.random.randint(1, 3000) 
        }
        customers.append(customer)
    return pd.DataFrame(customers)

def generate_base_transactions(customers_df, n_trans):
    """
    Generates 'Normal' behavior.
    """
    transactions = []
    customer_ids = customers_df['customer_id'].tolist()
    
    for _ in range(n_trans):
        cust_id = random.choice(customer_ids)
        # Get customer details for context
        cust_row = customers_df[customers_df['customer_id'] == cust_id].iloc[0]
        current_balance = cust_row['account_balance']
        
        # Normal Amount: Small, usually < 10% of balance
        amount = round(np.random.exponential(100), 2)
        if amount > current_balance: amount = current_balance * 0.1 # Cap it
        
        pct_balance = round((amount / current_balance) * 100, 2) if current_balance > 0 else 0

        tx = {
            "transaction_id": fake.uuid4(),
            "customer_id": cust_id,
            "timestamp": fake.date_time_between(start_date='-30d', end_date='now'),
            "amount": amount,
            "percent_balance_transferred": pct_balance,
            "beneficiary_id": fake.uuid4(),
            "beneficiary_bank_type": random.choice(BANK_TYPES), # Varied, mostly Traditional
            "device_id": cust_row['device_id'], 
            "location": fake.city(),
            "time_on_page_sec": np.random.randint(20, 60),
            "is_call_active": random.choice([True] + [False]*9), # 10% chance active call (multitasking)
            "is_new_beneficiary": random.choice([True] + [False]*4), 
            "is_fraud": 0 
        }
        transactions.append(tx)
    return pd.DataFrame(transactions)

def inject_scenario_a_panic_transfer(df, customers_df, n_fraud):
    """
    INJECTS FRAUD SCENARIO A: The Elderly Victim (Coercion)
    New Signals: Active Call = True, High % Balance
    """
    print(f"Injecting {n_fraud} 'Panic Transfer' fraud cases...")
    
    elderly_customers = customers_df[customers_df['age'] > 60]['customer_id'].tolist()
    
    fraud_cases = []
    for _ in range(n_fraud):
        cust_id = random.choice(elderly_customers)
        cust_row = customers_df[customers_df['customer_id'] == cust_id].iloc[0]
        current_balance = cust_row['account_balance']

        # Fraud Amount: High value, draining significant portion
        amount = round(np.random.uniform(2000, 9000), 2)
        if amount > current_balance: amount = current_balance * 0.95 # Drain 95%
        
        pct_balance = round((amount / current_balance) * 100, 2) if current_balance > 0 else 0

        tx = {
            "transaction_id": fake.uuid4(),
            "customer_id": cust_id,
            "timestamp": fake.date_time_between(start_date='-30d', end_date='now'),
            "amount": amount,
            "percent_balance_transferred": pct_balance,
            "beneficiary_id": fake.uuid4(),
            "beneficiary_bank_type": random.choice(['NeoBank', 'Crypto_Exchange']), # Scammers prefer these
            "device_id": cust_row['device_id'], 
            "location": fake.city(),
            "time_on_page_sec": np.random.randint(10, 40), # Panic
            "is_call_active": True, # CRITICAL: Victim is on phone with scammer
            "is_new_beneficiary": True, 
            "is_fraud": 1
        }
        fraud_cases.append(tx)
    
    return pd.concat([df, pd.DataFrame(fraud_cases)], ignore_index=True)

def inject_scenario_b_mule_account(df, customers_df, n_fraud):
    """
    INJECTS FRAUD SCENARIO B: The Mule (Laundering)
    New Signals: Bank Type = Crypto/Neo, Balance % = 100%
    """
    print(f"Injecting {n_fraud} 'Mule' fraud cases...")
    
    new_accounts = customers_df[customers_df['account_age_days'] < 30]['customer_id'].tolist()
    
    fraud_cases = []
    for _ in range(n_fraud):
        cust_id = random.choice(new_accounts) if new_accounts else customers_df['customer_id'].iloc[0]
        cust_row = customers_df[customers_df['customer_id'] == cust_id].iloc[0]
        
        # Mule logic: Money comes in, Money goes out immediately.
        # We simulate the 'out' transaction here.
        amount = round(np.random.uniform(500, 1500), 2) 
        # Mules often drain 100% of what they just received
        pct_balance = 99.9 

        tx = {
            "transaction_id": fake.uuid4(),
            "customer_id": cust_id,
            "timestamp": fake.date_time_between(start_date='-2d', end_date='now'),
            "amount": amount,
            "percent_balance_transferred": pct_balance,
            "beneficiary_id": fake.uuid4(),
            "beneficiary_bank_type": 'Crypto_Exchange', # Fast exit
            "device_id": fake.uuid4(), # Different device
            "location": fake.city(),
            "time_on_page_sec": np.random.randint(15, 45),
            "is_call_active": False, # Mule knows what they are doing, no one guiding them
            "is_new_beneficiary": True,
            "is_fraud": 1
        }
        fraud_cases.append(tx)
            
    return pd.concat([df, pd.DataFrame(fraud_cases)], ignore_index=True)

# --- EXECUTION ---
print("Generating Customers...")
customers = generate_customers(NUM_CUSTOMERS)

print("Generating Base Transactions...")
transactions = generate_base_transactions(customers, NUM_TRANSACTIONS)

# Inject Fraud
transactions = inject_scenario_a_panic_transfer(transactions, customers, n_fraud=100)
transactions = inject_scenario_b_mule_account(transactions, customers, n_fraud=50)

# Shuffle
transactions = transactions.sample(frac=1).reset_index(drop=True)

# Save
customers.to_csv("customers.csv", index=False)
transactions.to_csv("transactions.csv", index=False)
print("Enhanced Data saved to CSV.")
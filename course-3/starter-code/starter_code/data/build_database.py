#!/usr/bin/env python3
"""
Financial Database Builder

This script creates and populates the financial.db database with sample data.
Run this script only once to set up the database for the financial agent.

Usage: python build_database.py
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta
import random

def create_database():
    """Create the financial.db database with all necessary tables and sample data."""
    
    # Database path - handle both running from root and from data directory
    if os.path.exists('helper_modules'):
        # Running from project root
        db_path = os.path.join('data', 'financial.db')
        data_dir = 'data'
    else:
        # Running from data directory
        db_path = 'financial.db'
        data_dir = '.'
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        print(f"Removing existing database: {db_path}")
        os.remove(db_path)
    
    # Connect to database (this will create it)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Creating database tables...")
    
    # Create customers table
    cursor.execute('''
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            investment_profile TEXT CHECK(investment_profile IN ('conservative', 'moderate', 'aggressive')),
            risk_tolerance TEXT CHECK(risk_tolerance IN ('low', 'medium', 'high')),
            account_balance REAL DEFAULT 0.0,
            total_portfolio_value REAL DEFAULT 0.0
        )
    ''')
    
    # Create companies table
    cursor.execute('''
        CREATE TABLE companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            sector TEXT NOT NULL,
            market_cap REAL
        )
    ''')
    
    # Create portfolio_holdings table
    cursor.execute('''
        CREATE TABLE portfolio_holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            shares REAL NOT NULL,
            purchase_price REAL NOT NULL,
            current_value REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (symbol) REFERENCES companies(symbol)
        )
    ''')
    
    # Create market_data table
    cursor.execute('''
        CREATE TABLE market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            close_price REAL NOT NULL,
            volume INTEGER NOT NULL,
            market_cap REAL,
            date TEXT NOT NULL,
            FOREIGN KEY (symbol) REFERENCES companies(symbol)
        )
    ''')
    
    print("Tables created successfully!")
    
    # Insert sample companies
    print("Inserting company data...")
    companies_data = [
        ('AAPL', 'Apple Inc.', 'Technology', 3000000000000),
        ('GOOGL', 'Alphabet Inc.', 'Technology', 2000000000000),
        ('TSLA', 'Tesla Inc.', 'Automotive', 800000000000)
    ]
    
    cursor.executemany('''
        INSERT INTO companies (symbol, name, sector, market_cap)
        VALUES (?, ?, ?, ?)
    ''', companies_data)
    
    # Insert sample customers
    print("Inserting customer data...")
    customers_data = [
        ('John', 'Smith', 'john.smith@email.com', '555-0101', 'moderate', 'medium', 50000.0, 75000.0),
        ('Sarah', 'Johnson', 'sarah.johnson@email.com', '555-0102', 'aggressive', 'high', 100000.0, 150000.0),
        ('Michael', 'Brown', 'michael.brown@email.com', '555-0103', 'conservative', 'low', 25000.0, 30000.0),
        ('Emily', 'Davis', 'emily.davis@email.com', '555-0104', 'moderate', 'medium', 75000.0, 90000.0),
        ('David', 'Wilson', 'david.wilson@email.com', '555-0105', 'aggressive', 'high', 200000.0, 250000.0),
        ('Lisa', 'Anderson', 'lisa.anderson@email.com', '555-0106', 'conservative', 'low', 40000.0, 45000.0),
        ('Robert', 'Taylor', 'robert.taylor@email.com', '555-0107', 'moderate', 'medium', 80000.0, 95000.0),
        ('Jennifer', 'Martinez', 'jennifer.martinez@email.com', '555-0108', 'aggressive', 'high', 150000.0, 180000.0),
        ('Christopher', 'Garcia', 'christopher.garcia@email.com', '555-0109', 'conservative', 'low', 35000.0, 40000.0),
        ('Amanda', 'Rodriguez', 'amanda.rodriguez@email.com', '555-0110', 'moderate', 'medium', 60000.0, 70000.0)
    ]
    
    cursor.executemany('''
        INSERT INTO customers (first_name, last_name, email, phone, investment_profile, risk_tolerance, account_balance, total_portfolio_value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', customers_data)
    
    # Insert sample portfolio holdings
    print("Inserting portfolio holdings...")
    portfolio_data = [
        # John Smith (customer_id: 1)
        (1, 'AAPL', 50.0, 150.0, 8750.0),
        (1, 'GOOGL', 20.0, 2500.0, 3200.0),
        
        # Sarah Johnson (customer_id: 2)
        (2, 'AAPL', 100.0, 145.0, 17500.0),
        (2, 'TSLA', 75.0, 800.0, 15750.0),
        (2, 'GOOGL', 40.0, 2400.0, 6400.0),
        
        # Michael Brown (customer_id: 3)
        (3, 'AAPL', 25.0, 160.0, 4375.0),
        (3, 'GOOGL', 10.0, 2600.0, 1600.0),
        
        # Emily Davis (customer_id: 4)
        (4, 'AAPL', 75.0, 155.0, 13125.0),
        (4, 'TSLA', 30.0, 850.0, 6300.0),
        
        # David Wilson (customer_id: 5)
        (5, 'AAPL', 200.0, 140.0, 35000.0),
        (5, 'GOOGL', 80.0, 2300.0, 12800.0),
        (5, 'TSLA', 150.0, 750.0, 31500.0),
        
        # Lisa Anderson (customer_id: 6)
        (6, 'AAPL', 30.0, 165.0, 5250.0),
        (6, 'GOOGL', 15.0, 2700.0, 2400.0),
        
        # Robert Taylor (customer_id: 7)
        (7, 'AAPL', 80.0, 148.0, 14000.0),
        (7, 'TSLA', 40.0, 820.0, 8400.0),
        
        # Jennifer Martinez (customer_id: 8)
        (8, 'AAPL', 150.0, 142.0, 26250.0),
        (8, 'GOOGL', 60.0, 2450.0, 9600.0),
        (8, 'TSLA', 100.0, 780.0, 21000.0),
        
        # Christopher Garcia (customer_id: 9)
        (9, 'AAPL', 20.0, 170.0, 3500.0),
        (9, 'GOOGL', 8.0, 2800.0, 1280.0),
        
        # Amanda Rodriguez (customer_id: 10)
        (10, 'AAPL', 60.0, 152.0, 10500.0),
        (10, 'TSLA', 25.0, 860.0, 5250.0)
    ]
    
    cursor.executemany('''
        INSERT INTO portfolio_holdings (customer_id, symbol, shares, purchase_price, current_value)
        VALUES (?, ?, ?, ?, ?)
    ''', portfolio_data)
    
    # Insert sample market data (recent dates)
    print("Inserting market data...")
    base_date = datetime.now() - timedelta(days=30)
    market_data = []
    
    # Generate 30 days of market data for each symbol
    for i in range(30):
        current_date = (base_date + timedelta(days=i)).strftime('%Y-%m-%d')
        
        # AAPL data
        aapl_price = 175.0 + random.uniform(-10, 10)
        aapl_volume = random.randint(50000000, 100000000)
        market_data.append(('AAPL', aapl_price, aapl_volume, 3000000000000, current_date))
        
        # GOOGL data
        googl_price = 160.0 + random.uniform(-15, 15)
        googl_volume = random.randint(20000000, 40000000)
        market_data.append(('GOOGL', googl_price, googl_volume, 2000000000000, current_date))
        
        # TSLA data
        tsla_price = 210.0 + random.uniform(-20, 20)
        tsla_volume = random.randint(30000000, 80000000)
        market_data.append(('TSLA', tsla_price, tsla_volume, 800000000000, current_date))
    
    cursor.executemany('''
        INSERT INTO market_data (symbol, close_price, volume, market_cap, date)
        VALUES (?, ?, ?, ?, ?)
    ''', market_data)
    
    # Commit changes
    conn.commit()
    
    # Verify data insertion
    print("\nVerifying data insertion:")
    
    cursor.execute("SELECT COUNT(*) FROM customers")
    customer_count = cursor.fetchone()[0]
    print(f"Customers inserted: {customer_count}")
    
    cursor.execute("SELECT COUNT(*) FROM companies")
    company_count = cursor.fetchone()[0]
    print(f"Companies inserted: {company_count}")
    
    cursor.execute("SELECT COUNT(*) FROM portfolio_holdings")
    holdings_count = cursor.fetchone()[0]
    print(f"Portfolio holdings inserted: {holdings_count}")
    
    cursor.execute("SELECT COUNT(*) FROM market_data")
    market_count = cursor.fetchone()[0]
    print(f"Market data records inserted: {market_count}")
    
    # Close connection
    conn.close()
    
    print(f"\n✅ Database created successfully at: {db_path}")
    print("The financial agent is now ready to use!")

def main():
    """Main function to create the database."""
    print("Financial Database Builder")
    print("=" * 50)
    
    # Validate that we're in the right directory or data directory
    if not (os.path.exists('../helper_modules') or os.path.exists('helper_modules')):
        print("❌ Error: Please run this script from the project root directory")
        print("   (where helper_modules/ is located) or from the data/ directory")
        print("   Usage: python build_database.py")
        sys.exit(1)
    
    # Confirm with user before proceeding
    if os.path.exists('data/financial.db'):
        response = input("Database already exists. Do you want to recreate it? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Database creation cancelled.")
            sys.exit(0)
    
    try:
        create_database()
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

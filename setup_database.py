#!/usr/bin/env python3
import sqlite3
import argparse
import os
import json
from datetime import datetime, timedelta

def setup_database(db_path="writing_assistant.db", sample_data=False):
    """
    Set up the SQLite database for the writing assistant app.
    
    Args:
        db_path: Path to the database file
        sample_data: Whether to add sample data
    """
    # Create database connection
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Create users table (for future login functionality)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')
    
    # Create publications table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS publications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    ''')
    
    # Insert default publications
    publications = ["Daily Wealth Wire", "Weekly Investment Ideas", "Wealth Focus"]
    for pub in publications:
        cursor.execute('''
        INSERT OR IGNORE INTO publications (name) VALUES (?)
        ''', (pub,))
    
    # Create drafts table with memory_context field
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS drafts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT,
        publication_id INTEGER,
        user_id INTEGER,
        draft_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        approved BOOLEAN DEFAULT 0,
        memory_context TEXT,
        FOREIGN KEY (publication_id) REFERENCES publications (id),
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Add sample data if requested
    if sample_data:
        add_sample_data(conn, cursor)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database setup complete: {db_path}")
    if sample_data:
        print("Sample data added")

def add_sample_data(conn, cursor):
    """Add sample data to the database"""
    # Get publication IDs
    cursor.execute("SELECT id, name FROM publications")
    publications = {row['name']: row['id'] for row in cursor.fetchall()}
    
    # Sample dates
    today = datetime.now().date()
    yesterday = (datetime.now() - timedelta(days=1)).date()
    tomorrow = (datetime.now() + timedelta(days=1)).date()
    
    # Sample memory context (simplified for demonstration)
    sample_memory = {
        "messages": [
            {
                "type": "SystemMessage",
                "content": "You are an expert investment writer. You provide insightful analysis and clear recommendations.",
                "additional_kwargs": {},
                "timestamp": datetime.now().isoformat()
            },
            {
                "type": "HumanMessage",
                "content": "Please help me draft content for an investment publication.",
                "additional_kwargs": {},
                "timestamp": datetime.now().isoformat()
            },
            {
                "type": "AIMessage",
                "content": "I'll help you create a comprehensive investment draft. What specific topics would you like to focus on?",
                "additional_kwargs": {},
                "timestamp": datetime.now().isoformat()
            }
        ],
        "metadata": {
            "publication_type": "Sample Publication",
            "draft_date": today.isoformat(),
            "creation_timestamp": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "market_data": {},
            "topics": ["Technology", "Healthcare"],
            "recommendations": ["AAPL", "MSFT", "JNJ"]
        },
        "graph_state": {}
    }
    
    # Sample drafts
    sample_drafts = [
        {
            "title": "Tech Sector Analysis",
            "content": """# Tech Sector Analysis
            
## Market Overview
The technology sector continues to show resilience despite broader market volatility. Key players like Apple and Microsoft have reported strong earnings, driving positive sentiment.

## Investment Opportunities
- **Cloud Computing**: Azure and AWS continue to dominate
- **Semiconductors**: Supply chain issues are gradually resolving
- **AI Integration**: Companies implementing AI solutions are seeing productivity gains

## Recommendations
1. **Long-term holds**: AAPL, MSFT, GOOGL
2. **Emerging opportunities**: NVDA, AMD
3. **Speculative plays**: Smaller AI-focused startups

## Risk Assessment
While tech valuations remain high, the sector's growth prospects justify premium multiples for quality companies with strong cash flows.
            """,
            "publication_id": publications["Daily Wealth Wire"],
            "draft_date": today.isoformat(),
            "memory_context": json.dumps(sample_memory)
        },
        {
            "title": "Healthcare Investment Outlook",
            "content": """# Healthcare Investment Outlook
            
## Sector Performance
Healthcare has outperformed the broader market this quarter, with pharmaceuticals and medical devices leading the way.

## Regulatory Environment
Recent policy changes have created a more favorable environment for drug pricing, benefiting major pharmaceutical companies.

## Growth Areas
- **Telemedicine**: Continued adoption post-pandemic
- **Biotechnology**: mRNA platforms expanding beyond vaccines
- **Medical Devices**: Minimally invasive surgical tools

## Top Picks
1. Johnson & Johnson (JNJ)
2. UnitedHealth Group (UNH)
3. Moderna (MRNA)

## Conclusion
The healthcare sector offers both defensive characteristics and growth potential, making it an attractive area for balanced portfolios.
            """,
            "publication_id": publications["Weekly Investment Ideas"],
            "draft_date": yesterday.isoformat(),
            "memory_context": json.dumps(sample_memory)
        },
        {
            "title": "Renewable Energy Focus",
            "content": """# Renewable Energy Focus
            
## Market Transformation
The energy sector is undergoing a significant transformation as renewable sources gain market share and traditional energy companies pivot toward greener alternatives.

## Investment Thesis
1. **Policy Tailwinds**: Government incentives and regulations favor clean energy
2. **Cost Competitiveness**: Solar and wind have reached grid parity in many markets
3. **Corporate Commitments**: Major companies pledging carbon neutrality

## Key Players
- **Utilities**: NextEra Energy (NEE)
- **Solar**: First Solar (FSLR)
- **Wind**: Vestas Wind Systems (VWS)
- **Diversified**: Brookfield Renewable (BEP)

## Risk Factors
- Intermittency challenges
- Supply chain constraints
- Interest rate sensitivity

## Long-term Outlook
Despite near-term volatility, the secular trend toward renewable energy presents a multi-decade investment opportunity.
            """,
            "publication_id": publications["Wealth Focus"],
            "draft_date": tomorrow.isoformat(),
            "memory_context": json.dumps(sample_memory)
        }
    ]
    
    # Insert sample drafts
    for draft in sample_drafts:
        cursor.execute("""
        INSERT INTO drafts (title, content, publication_id, draft_date, memory_context)
        VALUES (?, ?, ?, ?, ?)
        """, (
            draft["title"],
            draft["content"],
            draft["publication_id"],
            draft["draft_date"],
            draft["memory_context"]
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up the database for the writing assistant app")
    parser.add_argument("--db", default="writing_assistant.db", help="Path to the database file")
    parser.add_argument("--sample-data", action="store_true", help="Add sample data to the database")
    
    args = parser.parse_args()
    setup_database(args.db, args.sample_data)

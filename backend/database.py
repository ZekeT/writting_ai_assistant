import sqlite3
import os
from datetime import datetime
import json
import threading


class Database:
    def __init__(self, db_path="writing_assistant.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.local = threading.local()
        self.create_tables()

    def get_connection(self):
        """Get a thread-local connection to the database"""
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            self.local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False)
            self.local.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return self.local.conn

    def get_cursor(self):
        """Get a cursor from the thread-local connection"""
        conn = self.get_connection()
        return conn.cursor()

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Users table (for future login functionality)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
            ''')

            # Publications table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS publications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
            ''')

            # Insert default publications
            publications = ["Daily Wealth Wire",
                            "Weekly Investment Ideas", "Wealth Focus"]
            for pub in publications:
                cursor.execute('''
                INSERT OR IGNORE INTO publications (name) VALUES (?)
                ''', (pub,))

            # Drafts table
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

            conn.commit()
        except sqlite3.Error as e:
            print(f"Table creation error: {e}")
            conn.rollback()
            raise

    def close(self):
        """Close the database connection for the current thread"""
        if hasattr(self.local, 'conn') and self.local.conn:
            self.local.conn.close()
            self.local.conn = None

    def get_publications(self):
        """Get all publications"""
        try:
            cursor = self.get_cursor()
            cursor.execute("SELECT * FROM publications")
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error fetching publications: {e}")
            return []

    def get_drafts_by_date_and_publication(self, publication_id, draft_date, user_id=None):
        """Get drafts filtered by date and publication"""
        try:
            cursor = self.get_cursor()
            query = """
            SELECT d.*, p.name as publication_name 
            FROM drafts d
            JOIN publications p ON d.publication_id = p.id
            WHERE d.publication_id = ? AND d.draft_date = ?
            """
            params = [publication_id, draft_date]

            if user_id:
                query += " AND d.user_id = ?"
                params.append(user_id)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error fetching drafts: {e}")
            return []

    def get_draft_by_id(self, draft_id):
        """Get a specific draft by ID"""
        try:
            cursor = self.get_cursor()
            cursor.execute("""
            SELECT d.*, p.name as publication_name 
            FROM drafts d
            JOIN publications p ON d.publication_id = p.id
            WHERE d.id = ?
            """, (draft_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as e:
            print(f"Error fetching draft: {e}")
            return None

    def create_draft(self, title, publication_id, draft_date, user_id=None, content="", memory_context=None):
        """Create a new draft"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO drafts (title, content, publication_id, user_id, draft_date, memory_context)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (title, content, publication_id, user_id, draft_date,
                  json.dumps(memory_context) if memory_context else None))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error creating draft: {e}")
            conn.rollback()
            return None

    def update_draft(self, draft_id, title=None, content=None, approved=None, memory_context=None):
        """Update an existing draft"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            update_parts = []
            params = []

            if title is not None:
                update_parts.append("title = ?")
                params.append(title)

            if content is not None:
                update_parts.append("content = ?")
                params.append(content)

            if approved is not None:
                update_parts.append("approved = ?")
                params.append(1 if approved else 0)

            if memory_context is not None:
                update_parts.append("memory_context = ?")
                params.append(json.dumps(memory_context))

            # Always update the updated_at timestamp
            update_parts.append("updated_at = CURRENT_TIMESTAMP")

            if not update_parts:
                return False  # Nothing to update

            query = f"UPDATE drafts SET {', '.join(update_parts)} WHERE id = ?"
            params.append(draft_id)

            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error updating draft: {e}")
            conn.rollback()
            return False

    def delete_draft(self, draft_id):
        """Delete a draft"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM drafts WHERE id = ?", (draft_id,))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error deleting draft: {e}")
            conn.rollback()
            return False

    def get_publication_id_by_name(self, publication_name):
        """Get publication ID by name"""
        try:
            cursor = self.get_cursor()
            cursor.execute(
                "SELECT id FROM publications WHERE name = ?", (publication_name,))
            result = cursor.fetchone()
            return result['id'] if result else None
        except sqlite3.Error as e:
            print(f"Error fetching publication ID: {e}")
            return None


# Singleton pattern for database access
_db_instance = None
_db_lock = threading.Lock()


def get_db_instance():
    """Get the singleton database instance in a thread-safe manner"""
    global _db_instance
    with _db_lock:
        if _db_instance is None:
            _db_instance = Database()
    return _db_instance

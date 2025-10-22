import sqlite3
import datetime
from langchain.messages import HumanMessage, AIMessage

def init():
    conn = sqlite3.connect('store_chat.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS CHAT_HISTORY (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS DOCUMENTS (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              filename TEXT NOT NULL,
              content TEXT NOT NULL,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()


def insert_message(role, content):
    conn = sqlite3.connect('store_chat.db')
    c = conn.cursor()
    c.execute("""INSERT INTO CHAT_HISTORY (role, content, timestamp) VALUES (?,?,?)""",(role, content, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_messages(limit : int):
    conn = sqlite3.connect('store_chat.db')
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM CHAT_HISTORY ORDER BY timestamp ASC LIMIT ?",(limit,))
    rows = c.fetchall()
    conn.close()
    rows.reverse()  # chronological order

    messages = []
    for role, content, _ in rows:
        if role.lower() == "user":
            messages.append(HumanMessage(content=content))
        elif role.lower() in ("ai", "assistant"):
            messages.append(AIMessage(content=content))
    return messages

def clear_messages():
    conn = sqlite3.connect('store_chat.db')
    c = conn.cursor()
    c.execute("DELETE FROM CHAT_HISTORY")
    conn.commit()
    conn.close()

def insert_document(filename, content):
    conn = sqlite3.connect('store_chat.db')
    c = conn.cursor()
    c.execute("""INSERT INTO DOCUMENTS (filename, content, timestamp) VALUES (?,?,?)""",
              (filename, content, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_document_by_filename(filename):
    conn = sqlite3.connect('store_chat.db')
    c = conn.cursor()
    c.execute("SELECT content FROM DOCUMENTS WHERE filename = ?", (filename,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def fetch_all_documents():
    conn = sqlite3.connect('store_chat.db')
    c = conn.cursor()
    c.execute("SELECT filename FROM DOCUMENTS")
    rows = c.fetchall()
    conn.close()
    return rows

def fetch_last_user_question():
    conn = sqlite3.connect('store_chat.db')
    c = conn.cursor()
    c.execute("SELECT content FROM CHAT_HISTORY WHERE role = 'user' ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def clear_documents():
    conn = sqlite3.connect('store_chat.db')
    c = conn.cursor()
    c.execute("DELETE FROM DOCUMENTS")
    conn.commit()
    conn.close()
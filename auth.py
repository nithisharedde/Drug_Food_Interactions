"""
auth.py
--------
Lightweight authentication and history storage for the Drug-Food
Interaction app. Uses SQLite (built into Python, no extra dependency)
and salted PBKDF2 password hashing (no plaintext / hardcoded
credentials anywhere).

This replaces the old hardcoded admin/1234 check with real accounts
that users create themselves.
"""

import os
import re
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,20}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Connection / schema
# ---------------------------------------------------------------------------
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist yet. Safe to call on every run."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            drug TEXT NOT NULL,
            food TEXT NOT NULL,
            interaction TEXT NOT NULL,
            reason TEXT,
            drug_taste TEXT,
            food_taste TEXT,
            confidence REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str, salt=None):
    """PBKDF2-HMAC-SHA256 with a random 16-byte salt. No plaintext storage."""
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib_pbkdf2(password, salt)
    return salt, pwd_hash


def hashlib_pbkdf2(password: str, salt: str) -> str:
    import hashlib

    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 200_000
    ).hex()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def valid_username(username: str) -> bool:
    return bool(USERNAME_RE.match(username or ""))


def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email or ""))


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------
def create_user(username: str, email: str, password: str):
    """Returns (success: bool, message: str)."""
    username = (username or "").strip()
    email = (email or "").strip().lower()

    if not valid_username(username):
        return False, "Username must be 3-20 characters: letters, numbers, or underscore."
    if not valid_email(email):
        return False, "Please enter a valid email address."
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters."

    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ? OR email = ?", (username, email)
    ).fetchone()
    if existing:
        conn.close()
        return False, "That username or email is already registered."

    salt, pwd_hash = hash_password(password)
    conn.execute(
        "INSERT INTO users (username, email, salt, password_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (username, email, salt, pwd_hash, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return True, "Account created! You can log in now."


def verify_user(username: str, password: str):
    """Returns (success: bool, user_row_as_dict_or_None)."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", ((username or "").strip(),)
    ).fetchone()
    conn.close()

    if row is None:
        return False, None

    check_hash = hashlib_pbkdf2(password or "", row["salt"])
    if check_hash == row["password_hash"]:
        return True, dict(row)
    return False, None


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def save_history(user_id, drug, food, interaction, reason, drug_taste, food_taste, confidence):
    conn = get_connection()
    conn.execute(
        """INSERT INTO history
           (user_id, drug, food, interaction, reason, drug_taste, food_taste, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            drug,
            food,
            interaction,
            reason,
            drug_taste,
            food_taste,
            confidence,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_history(user_id, limit=100):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_history(user_id):
    conn = get_connection()
    conn.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
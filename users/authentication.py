import hashlib
import hmac
import secrets
import sqlite3
from users.models import User


def _hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1).hex()


def register(conn, username, password):
    username = username.strip().casefold()
    if not 1 <= len(username) <= 64 or not 12 <= len(password) <= 256:
        raise ValueError('Username required; password must have 12–256 characters.')
    salt = secrets.token_bytes(16)
    try:
        cur = conn.execute('INSERT INTO users(username,password) VALUES (?,?)', (username, salt.hex()+':'+_hash(password,salt)))
    except sqlite3.IntegrityError as exc:
        raise ValueError('Account could not be created.') from exc
    return User(cur.lastrowid, username)


def authenticate(conn, username, password):
    if len(password)>256 or len(username)>64:
        return None
    row = conn.execute('SELECT * FROM users WHERE username=?', (username.strip().casefold(),)).fetchone()
    salt, expected = row['password'].split(':') if row else ('00'*16, '0'*128)
    valid = hmac.compare_digest(_hash(password, bytes.fromhex(salt)), expected)
    return User(row['id'],row['username']) if row and valid else None

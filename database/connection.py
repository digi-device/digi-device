from contextlib import contextmanager
from pathlib import Path
import sqlite3
from config import DATABASE

@contextmanager
def connect(path=DATABASE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        conn.executescript(Path(__file__).with_name('schema.sql').read_text())
        with conn:
            yield conn
    finally:
        conn.close()

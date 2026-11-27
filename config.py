from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
DATABASE = Path(os.environ.get('DIGI_DB', ROOT / 'data' / 'research.sqlite3'))

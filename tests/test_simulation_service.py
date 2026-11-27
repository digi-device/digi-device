import unittest
import tempfile
from pathlib import Path
from database.connection import connect
from interface.simulation_service import save_experiment, run_experiment

class TestSimulationService(unittest.TestCase):
    def test_invalid_request(self):
        with self.assertRaises(ValueError):
            run_experiment('not a patient',24,1,1,False)
    def test_graph_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.sqlite3'
            with connect(path) as conn:
                uid=conn.execute("INSERT INTO users(username,password) VALUES ('test','unused')").lastrowid
                rid=save_experiment(conn,uid,dict(patient='adult#001',hours=24),b'example')
            with connect(path) as conn:
                self.assertEqual(conn.execute('SELECT png FROM simulation_graphs WHERE run_id=?',(rid,)).fetchone()[0],b'example')

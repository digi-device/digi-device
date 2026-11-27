import unittest,tempfile
from pathlib import Path
from database.connection import connect
from users.authentication import register,authenticate
from personalization.observations import record
from personalization.sensitivity_history import history
from personalization.notes import add_note,notes
from personalization.profile_versions import save,versions

class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'test.db'
    def test_accounts_and_persistence(self):
        with connect(self.path) as c:
            a=register(c,'alice','long password 123');b=register(c,'bob','long password 456');record(c,a.id,1,2);add_note(c,a.id,'synthetic note')
        with connect(self.path) as c:
            self.assertEqual(authenticate(c,'ALICE','long password 123'),a);self.assertIsNone(authenticate(c,'alice','wrong'));self.assertEqual(len(history(c,a.id)),1);self.assertEqual(len(notes(c,a.id)),1);self.assertFalse(history(c,b.id));self.assertFalse(notes(c,b.id))
    def test_versions(self):
        with connect(self.path) as c:
            a=register(c,'alice','long password 123')
            for i in range(12):record(c,a.id,1 if i<4 else 2,1)
            save(c,a.id);save(c,a.id);self.assertEqual(len(versions(c,a.id)),1);self.assertEqual(versions(c,a.id)[0]['insulin_index'],2)
    def test_invalid(self):
        with connect(self.path) as c:
            a=register(c,'alice','long password 123')
            with self.assertRaises(ValueError):record(c,a.id,float('nan'),1)
            with self.assertRaises(ValueError):save(c,a.id)
            with self.assertRaises(ValueError):register(c,'alice','long password 123')

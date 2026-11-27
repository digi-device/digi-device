import json
from personalization.profile_evaluation import evaluate
from personalization.sensitivity_history import history
from users.repository import audit

def save(conn,uid):
    rows=history(conn,uid)
    result=evaluate(rows)
    if result is None:
        raise ValueError('Twelve valid synthetic observations required.')
    values=[v['candidate'] if v['improved'] else v['baseline'] for v in (result['insulin_index'],result['glucagon_index'])]
    cur=conn.execute('INSERT OR IGNORE INTO research_profiles(user_id,observation_id,insulin_index,glucagon_index,evidence) VALUES (?,?,?,?,?)',(uid,rows[-1]['id'],*values,json.dumps(result)))
    if cur.rowcount:
        audit(conn,uid,'profile_saved',str(cur.lastrowid))

def versions(conn,uid):
    return conn.execute('SELECT * FROM research_profiles WHERE user_id=? ORDER BY id DESC',(uid,)).fetchall()

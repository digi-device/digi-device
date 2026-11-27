def get_user(conn, uid):
    return conn.execute('SELECT id,username FROM users WHERE id=?',(uid,)).fetchone()


def audit(conn, uid, event, detail=''):
    conn.execute('INSERT INTO audit(user_id,event,detail) VALUES (?,?,?)',(uid,event,detail))

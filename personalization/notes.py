def add_note(conn, uid, body):
    body=body.strip()
    if not 1<=len(body)<=2000:
        raise ValueError('Note must have 1–2000 characters.')
    conn.execute('INSERT INTO notes(user_id,body) VALUES (?,?)',(uid,body))

def notes(conn,uid):
    return conn.execute('SELECT * FROM notes WHERE user_id=? ORDER BY id DESC',(uid,)).fetchall()

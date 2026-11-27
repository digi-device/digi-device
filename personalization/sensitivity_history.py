def history(conn, uid):
    return conn.execute('SELECT * FROM observations WHERE user_id=? ORDER BY id',(uid,)).fetchall()

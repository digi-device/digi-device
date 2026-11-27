import math

def record(conn, uid, insulin, glucagon, note=''):
    values = tuple(float(x) for x in (insulin, glucagon))
    if not all(math.isfinite(x) and 0<x<=100 for x in values) or len(note)>2000:
        raise ValueError('Invalid synthetic observation.')
    conn.execute('INSERT INTO observations(user_id,insulin_index,glucagon_index,note) VALUES (?,?,?,?)',(uid,*values,note))

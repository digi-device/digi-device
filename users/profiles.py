from personalization.profile_versions import versions

def current(conn, uid):
    rows=versions(conn,uid)
    return rows[0] if rows else None

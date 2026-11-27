from statistics import median

def trends(rows):
    if len(rows)<6:
        return {}
    return {field:(median(r[field] for r in rows[-6:-3]),median(r[field] for r in rows[-3:])) for field in ('insulin_index','glucagon_index')}

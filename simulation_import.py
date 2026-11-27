"""Import a synthetic simulator CSV for offline descriptive review."""
import argparse,csv,json,math,getpass
from database.connection import connect
from users.authentication import authenticate

def summarize(path):
    with open(path,newline='') as f:
        vals=[float(row['BG']) for row in csv.DictReader(f)]
    if not vals or not all(math.isfinite(v) for v in vals):
        raise ValueError('Invalid or empty BG series')
    return {'samples':len(vals),'mean':sum(vals)/len(vals),'minimum':min(vals),'maximum':max(vals)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('csv');p.add_argument('username');args=p.parse_args()
    with connect() as conn:
        user=authenticate(conn,args.username,getpass.getpass())
        if not user: raise SystemExit('Invalid credentials')
        conn.execute('INSERT INTO simulation_runs(user_id,label,summary) VALUES (?,?,?)',(user.id,args.csv,json.dumps(summarize(args.csv))))
if __name__=='__main__':main()

import secrets
import json
from io import BytesIO
from flask import Blueprint, request, session, redirect, render_template, abort, g, send_file
from database.connection import connect
from users.authentication import register, authenticate
from users.repository import get_user
from users.profiles import current
from personalization.observations import record
from personalization.sensitivity_history import history
from personalization.notes import add_note, notes
from personalization.trend_analysis import trends
from personalization.profile_evaluation import evaluate
from personalization.profile_versions import save, versions
from interface.simulation_service import PATIENTS, run_experiment, save_experiment

bp=Blueprint('portal',__name__,template_folder='templates',static_folder='static')

@bp.before_request
def guard():
    if request.method=='POST' and not secrets.compare_digest(session.get('csrf','!'),request.form.get('csrf','')):
        abort(400)
    session.setdefault('csrf',secrets.token_hex(32))
    g.uid=session.get('uid')
    if request.endpoint not in ('portal.auth','portal.static'):
        with connect() as conn:
            if not g.uid or not get_user(conn,g.uid):
                return redirect('/login')

@bp.route('/login',methods=['GET','POST'])
@bp.route('/register',methods=['GET','POST'])
def auth():
    mode='Register' if request.path=='/register' else 'Login'
    error=''
    if request.method=='POST':
        try:
            with connect() as conn:
                fn=register if mode=='Register' else authenticate
                user=fn(conn,request.form.get('username',''),request.form.get('password',''))
                if not user: raise ValueError('Invalid credentials.')
            session.clear()
            session['uid']=user.id
            return redirect('/')
        except ValueError as exc:
            error=str(exc)
    return render_template('auth.html',mode=mode,error=error)

@bp.post('/logout')
def logout():
    session.clear()
    return redirect('/login')

@bp.get('/')
@bp.get('/profile')
@bp.get('/history')
@bp.get('/notes')
def page():
    with connect() as conn:
        rows=history(conn,g.uid)
        return render_template((request.path.strip('/') or 'dashboard')+'.html',user=get_user(conn,g.uid),rows=rows,profile=current(conn,g.uid),analysis=evaluate(rows),trends=trends(rows),notes=notes(conn,g.uid),versions=versions(conn,g.uid),runs=conn.execute('SELECT * FROM simulation_runs WHERE user_id=? ORDER BY id DESC',(g.uid,)).fetchall())

@bp.post('/observations')
@bp.post('/notes')
@bp.post('/snapshot')
def update():
    try:
        with connect() as conn:
            if request.path=='/observations': record(conn,g.uid,request.form.get('insulin',''),request.form.get('glucagon',''),request.form.get('note',''))
            elif request.path=='/notes': add_note(conn,g.uid,request.form.get('body',''))
            else: save(conn,g.uid)
    except ValueError as exc:
        abort(400,description=str(exc))
    return redirect({'/observations':'/history','/notes':'/notes','/snapshot':'/profile'}[request.path])


@bp.get('/simulation')
def simulation_page():
    return show_simulation()


def show_simulation(error='', selected=None):
    with connect() as conn:
        runs = conn.execute('''SELECT r.*, g.run_id AS graph_id FROM simulation_runs r
            LEFT JOIN simulation_graphs g ON g.run_id=r.id
            WHERE r.user_id=? ORDER BY r.id DESC''', (g.uid,)).fetchall()
        parsed = [dict(id=r['id'], label=r['label'], recorded_at=r['recorded_at'],
                       graph_id=r['graph_id'], summary=json.loads(r['summary'])) for r in runs]
        return render_template('simulation.html', user=get_user(conn,g.uid),
                               patients=PATIENTS, runs=parsed, error=error, selected=selected)


@bp.post('/simulation/run')
def launch_simulation():
    try:
        patient = request.form.get('patient','')
        hours = int(request.form.get('hours','24'))
        seed = int(request.form.get('seed','1'))
        factor = float(request.form.get('factor','1'))
        randomized = request.form.get('reporting') == 'random'
        summary, image = run_experiment(patient,hours,seed,factor,randomized)
        with connect() as conn:
            run_id = save_experiment(conn,g.uid,summary,image)
    except (ValueError, RuntimeError, OSError) as exc:
        return show_simulation(error=str(exc)), 400
    return redirect(f'/simulation#run-{run_id}')


@bp.get('/simulation/graph/<int:run_id>')
def simulation_graph(run_id):
    with connect() as conn:
        row = conn.execute('''SELECT g.png FROM simulation_graphs g
            JOIN simulation_runs r ON r.id=g.run_id
            WHERE r.id=? AND r.user_id=?''', (run_id,g.uid)).fetchone()
    if row is None:
        abort(404)
    response = send_file(BytesIO(row['png']), mimetype='image/png', max_age=0)
    response.headers['Cache-Control'] = 'private, no-store'
    return response

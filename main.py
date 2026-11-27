import os
import secrets
from flask import Flask
from interface.routes import bp

def create_app():
    app=Flask(__name__,static_folder=None)
    app.config.update(SECRET_KEY=os.environ.get('DIGI_SECRET_KEY') or secrets.token_hex(32),SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Strict',MAX_CONTENT_LENGTH=16384)
    app.register_blueprint(bp)
    return app

if __name__=='__main__':
    create_app().run(host='127.0.0.1',port=5000,debug=False)

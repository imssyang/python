# FLASK_APP=xxx FLASK_DEBUG=1 flask run --host 0.0.0.0 --port 8008
import secrets
from flask import Flask
from flask import make_response
from flask import render_template
from flask import request
from flask import session
from flask import abort, redirect, url_for
from markupsafe import escape
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.secret_key = secrets.token_hex()

app.wsgi_app = ProxyFix(app.wsgi_app)

@app.route("/<name>")
def hello(name):
    return f"Hello, {escape(name)}!"

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return f'User {escape(username)}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return f'Post {post_id}'

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    # show the subpath after /path/
    return f'Subpath {escape(subpath)}'

@app.route('/projects/')
def projects():
    abort(401)
    return 'The project page'

@app.route('/about')
def about():
    username = request.cookies.get('username')
    resp = make_response(render_template(...))
    resp.set_cookie('username', 'the username')
    return resp

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('hello', name='ssy'))
    return '''
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''

@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/login_test', methods=['GET', 'POST'])
def login_test():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'],
                       request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password'
    return render_template('login.html', error=error)

@app.get('/login_get')
def login_get():
    return show_the_login_form()

@app.post('/login_post')
def login_post():
    return do_the_login()

@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'

@app.route('/hello/')
@app.route('/hello/<name>')
def hello_template(name=None):
    return render_template('hello.html', name=name)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save(f"uploads/{secure_filename(file.filename)}")

@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

@app.errorhandler(401)
def error_401(error):
    resp = make_response(render_template('error.html'), 401)
    resp.headers['X-Something'] = 'A value'
    return resp

@app.route("/me")
def me_api():
    user = get_current_user()
    # dict or list converted to a JSON response
    return {
        "username": user.username,
        "theme": user.theme,
        "image": url_for("user_image", filename=user.image),
    }

with app.test_request_context():
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))
    print(url_for('static', filename='style.css'))

with app.test_request_context('/hello', method='POST'):
    assert request.path == '/hello'
    assert request.method == 'POST'

if __name__ == '__main__':
    #app.run()
    app.wsgi_app.run()

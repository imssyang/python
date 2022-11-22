# https://uwsgi-docs.readthedocs.io/en/latest/WSGIquickstart.html
# uwsgi --http :5000 --wsgi-file wsgi.py
# uwsgi --http :5000 --wsgi-file wsgi.py --master --processes 4 --threads 2
# uwsgi --http :5000 --wsgi-file wsgi.py --master --processes 4 --threads 2 --stats :5001
# uwsgi --http-socket :5000 --chdir /opt/uwsgi --wsgi-file wsgi.py --master --processes 4 --threads 2 --stats :5001
def application(env, start_response):
    start_response("200 OK", [("Content-Type", "text/html")])
    return [b"Hello World"]

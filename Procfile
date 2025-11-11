web: gunicorn app:app
worker: python -c "from app import app, db; app.app_context().push(); db.create_all(); print('Database initialized')"

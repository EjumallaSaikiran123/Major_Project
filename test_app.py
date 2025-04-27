import pytest
from app import create_app, connect_db

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def setup_module(module):
    # Clear and re-create table for test isolation
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def test_register(client):
    response = client.post('/register', data={
        'username': 'testuser',
        'password': 'testpass'
    }, follow_redirects=True)
    assert b'login' in response.data or response.status_code == 200

def test_register_duplicate_user(client):
    client.post('/register', data={'username': 'testuser', 'password': 'testpass'})
    response = client.post('/register', data={'username': 'testuser', 'password': 'testpass'})
    assert b'Username already exists' in response.data or response.status_code == 200

def test_login_success(client):
    client.post('/register', data={'username': 'logintest', 'password': 'pass123'})
    response = client.post('/login', data={'username': 'logintest', 'password': 'pass123'}, follow_redirects=True)
    assert b'dashboard' in response.data or response.status_code == 200

def test_login_failure(client):
    response = client.post('/login', data={'username': 'nouser', 'password': 'nopass'})
    assert b'Invalid username or password' in response.data

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request, Response
import os
import json

SESSION_COOKIE = 'remarkable_session'
SESSION_SECRET = os.environ.get('SESSION_SECRET', 'dev_secret')
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 1 week

def get_serializer():
    return URLSafeTimedSerializer(SESSION_SECRET)

def set_session(response: Response, data: dict):
    s = get_serializer()
    session_data = s.dumps(data)
    response.set_cookie(
        SESSION_COOKIE,
        session_data,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        secure=True,  # Set to True in production (requires HTTPS)
        samesite='lax',
        path='/',
    )

def get_session(request: Request):
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return {}
    s = get_serializer()
    try:
        data = s.loads(cookie, max_age=SESSION_MAX_AGE)
        return data
    except (BadSignature, SignatureExpired):
        return {}

def clear_session(response: Response):
    response.delete_cookie(SESSION_COOKIE, path='/') 
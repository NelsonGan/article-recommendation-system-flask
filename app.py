from flask import Flask
from flask import request
from helper import get_recommended_posts_for

app = Flask(__name__)


@app.route('/api/recommended-posts', methods=['GET'])
def posts_recommend():
    user_id = int(request.args.get('user_id'))

    return get_recommended_posts_for(user_id)

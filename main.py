from flask import Flask, jsonify
from api.controllers.defaultController import defaultController
from routes import routes

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'message': 'Hello World'})

# Register the routes Blueprint
#app.register_blueprint(routes)

#default_controller = DefaultController()

if __name__ == '__main__':
    app.run(port=5000, debug=True)  # Set the desired port numbers
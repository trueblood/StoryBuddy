from flask import Flask, jsonify
from controllers.defaultController import DefaultController
from routes import routes

app = Flask(__name__)

# Register the routes Blueprint
app.register_blueprint(routes)

default_controller = DefaultController()

class DefaultController:
    def __init__(self):
        pass
    
    @app.route('/api/myroute', methods=['POST'])
    def my_route(self):
        # Handle the route logic here
        return jsonify({"message": "Received", "data": ""}), 200

if __name__ == '__main__':
    app.run(port=5000)  # Set the desired port numbers
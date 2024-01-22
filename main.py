from flask import Flask, jsonify, request
from flask_cors import CORS
from api.controllers.defaultController import defaultController
from routes import routes


app = Flask(__name__)

CORS(app, resources={r"/generate_story": {"origins": ["http://localhost:5001", "https://storybuddy.angrybuddy.com", "http://192.168.1.53:5070", "http://75.187.70.207:5070"]}})

@app.route('/')
def index():
    return jsonify({'message': 'server up and running'})

@app.route('/generate_story', methods=['POST'])
def generate_story_route():
    try:
        prompt = request.json.get('prompt')
        max_length = request.json.get('max_length')
        recaptcha_response = request.json.get('recaptcha_response')
        controller = defaultController()
        result = controller.generate_story(prompt, int(max_length), recaptcha_response)
        return jsonify(result)
        #return jsonify(prompt, max_length, recaptcha_response)
    except Exception as e:
        return jsonify({'error': str(e)})
    

# Register the routes Blueprint
#app.register_blueprint(routes)

#default_controller = DefaultController()

if __name__ == '__main__':
    app.run(port=5070, debug=True)  # Set the desired port numbers
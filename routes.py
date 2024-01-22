from flask import Blueprint, jsonify, request
from api.controllers.defaultController import defaultController as DefaultController

# Create a Blueprint for the routes
routes = Blueprint('routes', __name__)

# Create an instance of the DefaultController
default_controller = DefaultController()

# Define a route
@routes.route('/api/generate-story', methods=['POST'])
def generate_story():
    # Get the request data
    data = request.get_json()
    prompt = data.get('prompt')
    max_length = data.get('max_length')

    # Generate the story using the DefaultController
    generated_story = default_controller.generate_story(prompt, max_length)

    # Return the generated story as a JSON response
    return jsonify({'generated_story': generated_story})

@routes.route('/api/verify-recaptcha', methods=['POST'])
def verify_recaptcha():
    # Get the recaptcha response from the request
    recaptcha_response = request.form.get('recaptcha_response')

    # Verify the recaptcha using the DefaultController
    success = default_controller.verify_recaptcha(recaptcha_response)

    # Return the verification result as a JSON response
    return jsonify({'success': success})

# Add more routes as needed
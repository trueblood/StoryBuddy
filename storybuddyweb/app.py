from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Story Buddy')

@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    # Example API call
    response = requests.get('https://api.example.com/data')
    data = response.json()
    return jsonify(data)

@app.route('/history')
def history():
    return render_template('history.html', title='Story Buddy')

if __name__ == '__main__':
    host = 'localhost'  # Local host name
    port = 5001  # Port number
    app.run(host=host, port=port, debug=True)

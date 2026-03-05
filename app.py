from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def hello_world(path):
        data = request.stream.read()
        print("Path: " + path)
        print("Data: " + str(data))
        return "Hello, World!"

if __name__ == '__main__':
        app.run(host='0.0.0.0',port=9000)

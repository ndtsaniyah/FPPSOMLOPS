from flask import request, jsonify, render_template
from . import utils

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/label', methods=['POST'])
def label_data():
    data = request.json['data']
    labeled_data = utils.your_labeling_function(data)
    return jsonify(labeled_data)

if __name__ == '__main__':
    app.run(debug=True)


from flask import request, jsonify, render_template
from . import utils
from flask import request, jsonify, render_template, current_app as app
from model.labeling_model import label_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/label', methods=['POST'])
def label_data_route():
    data = request.json['data']
    labeled_data = utils.labeling_function(data)
    return jsonify(labeled_data)

if __name__ == '__main__':
    app.run(debug=True)

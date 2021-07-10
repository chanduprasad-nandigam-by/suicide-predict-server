import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin



app = Flask(__name__)
CORS(app)





incomes = [
    {'description': 'salary', 'amount': 5000}
]


@app.route('/')
def get_incomes():
  return jsonify(incomes)


@app.route('/', methods=['POST'])
def add_income():
  incomes.append(request.get_json())
  return jsonify(incomes), 201


@app.route('/predict', methods=['POST'])
def suicide_predictor():
  return jsonify(request.get_json()), 201


if __name__ == '__main__':
    app.run()

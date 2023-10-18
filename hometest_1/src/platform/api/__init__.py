import os

##########################################
#
# APP Initialization
#
###########################################
from flask import Flask
from flask_cors import CORS
from flask_restx import Api

app = Flask(__name__)
CORS(app)


##########################################
#
# DB and ORM Initialization section
#
###########################################


###########################################
#
# API routing section
#
###########################################
from .pdm import pdm_api

api = Api(app)
api.add_namespace(pdm_api)
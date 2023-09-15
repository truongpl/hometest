"""
Module: api.pdm.pdm

"""
##########################################
#
# Library import
#
###########################################
import os
from flask import request
from flask_restx import Namespace, Resource, fields
import json
import requests

##########################################
#
# Common import 
#
###########################################
from common_utils.log import logger
from common_utils.constants import HTTP_CODE, RETURN_CODE, MESSAGE_CODE

##########################################
#
# Module import 
#
###########################################
from services.pdm.service import PdmService

pdm_api = Namespace("pdm")
single_record = pdm_api .model('single_record', {
    "volt": fields.Float(required=True, description="Machine voltage"),
    "rotate": fields.Float(required=True, description="Machine rotate"),
    "pressure": fields.Float(required=True, description="Machine pressure"),
    "vibration": fields.Float(required=True, description="Machine vibration"),
    "age": fields.Float(required=True, description="Machine age"),
})
analyze_pdm = pdm_api.model('analyze_pdm', {
    "data": fields.Nested(single_record, as_list=True, required=True, description="Input list data"),
})

@pdm_api.route("/")
class Pdm(Resource):
    @pdm_api.doc("pdm",
                  description="Health check API",
                  responses={HTTP_CODE.OK: "OK"})
    def get(self):
        """
        Ping Check API
        """
        logger.info(" ".join([request.method, request.path]))

        return (
            {
                "result": RETURN_CODE.OK,
                "data": None
            },
            HTTP_CODE.OK
        )


    @pdm_api.doc("analyze pdm",
                  description="Predictive maintenance",
                  responses={HTTP_CODE.OK: "OK",
                            HTTP_CODE.BAD_REQUEST: "Invalid input for server",
                            HTTP_CODE.SERVER_ERROR: "Server error"})
    @pdm_api.expect(analyze_pdm)
    def post(self):
        """
        Receive Post data list
        """
        logger.info(" ".join([request.method, request.path]))

        req_json = request.get_json()
        logger.info(req_json)
        
        # Sanity check the key
        if "data" not in req_json.keys():
            return (
                {
                    "result": RETURN_CODE.NG,
                    "message": MESSAGE_CODE.BAD_MSG,
                    "data": None
                },
                HTTP_CODE.BAD_REQUEST
            )

        # Send to PdmService
        data = req_json["data"]
        if len(data) == 0:
            return (
                {
                    "result": RETURN_CODE.NG,
                    "message": MESSAGE_CODE.NULL_MSG,
                    "data": None
                },
                HTTP_CODE.BAD_REQUEST
            )

        pdm_service = PdmService()
        result = pdm_service.analyze(data)

        if result is not None:
            logger.info("[Pdm] Server predictions: {}".format(str(result)))
            return (
                {
                    "result": RETURN_CODE.OK,
                    "message": MESSAGE_CODE.OK_MSG,
                    "data": result
                },
                HTTP_CODE.OK
            )
        else:
            logger.error("[Pdm]: Server error")
            return (
                {
                    "result": RETURN_CODE.NG,
                    "message": MESSAGE_CODE.NG_MSG,
                    "data": None
                },
                HTTP_CODE.SERVER_ERROR
            )



"""
Module: common_utils.constants

"""
class HTTP_CODE():
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    SERVER_ERROR = 500
    BAD_GATEWAY = 503
    EXPIRED = 419

class RETURN_CODE():
    OK = 0
    NG = 1

class MESSAGE_CODE():
    BAD_MSG = "Need data keys in request"
    NULL_MSG = "data list is empty"
    OK_MSG = "OK"
    NG_MSG = "NG"

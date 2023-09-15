import os, time, json, re
import requests
import json
import uuid
import openai
openai.api_key=os.environ.get("LLM_KEY")

from common_utils.base_service import BaseService
class PdmService(BaseService):
    def set_up(self):
        # Setting up anything if any
        pass

    def tear_down(self):
        # Tear down everything: session, db connection blah blah blah
        pass

    def __init__(self):
        super().__init__()
        self.ft_model = os.environ.get("LLM_MODEL")
        self.system_prompt = """Giving input volt,rotate,pressure,vibration,age answer with number,number, first number is 0/1, second is regression in 0-23 range"""
        self.llm_pattern = r'^\d{1},\d{1}$'


    def analyze(self, data):
        m_volt = 0.0
        m_rotate = 0.0
        m_pressure = 0.0
        m_vibration = 0.0
        m_age = 0.0

        for i in data:
            m_volt += i["volt"]
            m_rotate += i["rotate"]
            m_pressure += i["pressure"]
            m_vibration += i["vibration"]
            m_age += i["age"]

        m_volt /= len(data)
        m_rotate /= len(data)
        m_pressure /= len(data)
        m_vibration /= len(data)
        m_age /= len(data)

        ### Prepare prompt message
        user_message = ",".join([str(m_volt),str(m_rotate),str(m_pressure),str(m_vibration),str(int(m_age))])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        completion = openai.ChatCompletion.create(
          model=self.ft_model,
          messages=messages
        )
            
        # Follow suggestion on: https://platform.openai.com/docs/guides/error-codes/api-errors
        try:
            resp_message = completion.choices[0].message["content"]
        except Exception as e: 
            logger.debug("[Pdm] Call to openai error ".join(str(e)))
            return None

        if re.match(self.llm_pattern, resp_message):
            return self.prepare_resp(resp_message)
        else:
            return None
        

    def prepare_resp(self, resp):
        result = None

        splitter = resp.split(",")
        is_mnt = int(splitter[0])
        mnt_time = int(splitter[1])
        
        if is_mnt == 0:
            result = {
                "pred_text": "Machine will works normally for the next 24h", # Put text here
                "pred_mnt": is_mnt,
                "pred_time": 0
            }
        else:
            result = {
                "pred_text": "Machine may have failure in next day at {}.00.00 ".format(splitter[1]),
                "pred_mnt": is_mnt,
                "pred_time": mnt_time
            }

        return result
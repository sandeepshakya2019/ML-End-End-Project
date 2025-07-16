import sys

def error_message_details(err, error_detail : sys):
    info = error_detail.exc_info()
    # exc_tb
    print(info)
    file_name = "file"
    line_number = "info.exc_tb"
    error = str(err)
    print(info)
    error_msg = f"Error Occured in python script name [{file_name}] line number [{line_number}] error msg is [{error}]"
    return error_msg

class CustomeException(Exception):
    def __init__(self, error_msg, error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_message_details(error_msg, error_detail=error_detail)
    
    def __str__(self):
        return self.error_msg
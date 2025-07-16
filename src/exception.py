import sys
from logger import logger

def error_message_details(err, error_detail: sys):
    # Unpack the exception details
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    
    # Extract file name and line number from the traceback
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error = str(err)
    
    error_msg = (
        f"\n\n❌❌❌ \n\nError occurred in Python script [{file_name}] "
        f"at line [{line_number}]: {error} \n\n❌❌❌\n"
    )
    
    return error_msg

class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)
        self.error_msg = error_message_details(error_msg, error_detail)
        logger.error(self.error_msg)  # Log the error (optional)
    
    def __str__(self):
        return self.error_msg

# Test block
if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)

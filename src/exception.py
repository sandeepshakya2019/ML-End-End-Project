# custom_exception.py
import sys
import traceback
from logger import logger

def error_message_details(err, error_detail: sys):
    # Unpack the exception info
    exc_type, exc_value, exc_tb = error_detail.exc_info()

    # Extract script name and line number
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error = str(err)

    # Optional: Full traceback
    formatted_traceback = traceback.format_exc()

    # Final error message
    error_msg = (
        f"\n\n❌❌❌ ERROR REPORT ❌❌❌\n"
        f"File       : {file_name}\n"
        f"Line       : {line_number}\n"
        f"Error      : {error}\n"
        f"Traceback  :\n{formatted_traceback}\n"
        f"❌❌❌ END ERROR ❌❌❌\n"
    )
    return error_msg

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error_msg = error_message_details(error, error_detail)
        logger.error(self.error_msg)  # Logs to file
    
    def __str__(self):
        return self.error_msg

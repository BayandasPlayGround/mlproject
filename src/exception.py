'''
We can define custom exceptions by creating a new class that inherits from the built-in Exception class. 
This allows us to create specific error types that can be caught and handled separately in our code. 
Here's an example of how to define a custom exception:
'''
import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info() # detail regarding the error
    error_message = f"Error occurred in script: {exc_tb.tb_frame.f_code.co_filename} at line number: {exc_tb.tb_lineno} error message: {str(error)}"
    return error_message

class CustomException(Exception): # inheriting the exception class
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    

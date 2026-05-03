import pandas as pd
import openpyxl
from src.logger import logger

def load_data(filepath):
    logger.info('Loading Dataset')
    try:
        data=pd.read_excel(filepath)
    except Exception as e:
        logger.error(f"Error Occured while reading file: {e}")
    return data
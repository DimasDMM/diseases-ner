import logging
import os
import pycurl
from io import BytesIO 
import re
import numpy as np

DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../data/"

def run(logging):
    logging.debug('Retrieving information from www.nhsinform.scot...')
    page_body = request()
    
    logging.debug('Parsing diseases...')
    diseases_list = parse_body(page_body)

    logging.debug('Saving file...')
    with open(DATA_PATH + 'diseases.txt', 'w') as fp:
        diseases_list = "\n".join(diseases_list)
        fp.write(diseases_list)
    
    logging.info('Done!')

def request():
    b_obj = BytesIO()
    curl = pycurl.Curl()

    curl.setopt(curl.URL, 'https://www.nhsinform.scot/illnesses-and-conditions/a-to-z')
    curl.setopt(curl.WRITEDATA, b_obj)
    curl.setopt(pycurl.SSL_VERIFYPEER, 0)
    curl.setopt(pycurl.SSL_VERIFYHOST, 0)
    curl.perform()
    curl.close()

    return b_obj.getvalue().decode('utf8')

def parse_body(page_body):
    page_body = re.findall(r'<h2 class="module__title">\s*(.+?)\r?\s*</h2>', page_body)
    page_body = [re.sub(r'&#39;s', '', d) for d in page_body]
    page_body = [re.sub(r'[:;].+$', '', d) for d in page_body]

    diseases_list = []
    for diseases_raw in page_body:
        diseases_raw = re.findall(r'([^\(\[]+)(?:[\[\(]([^\]\)]+?)[\]\)])?.*$', diseases_raw)[0]
        diseases_list = diseases_list + [d.strip() for d in diseases_raw if d.strip() != '']

    # Remove duplicates
    diseases_list = list(dict.fromkeys(diseases_list))
    diseases_list.sort()
    
    return diseases_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info('## WEB SCRAPPER ##')
    run(logging)

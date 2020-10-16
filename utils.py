from io import StringIO
from pathlib import Path

import numpy as np
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def convert_pdf_to_txt(pdf_path_string, all_texts):
    resource_manager = PDFResourceManager()
    returned_string = StringIO()
    codec = 'utf-8'
    layout_parameters = LAParams(all_texts=all_texts)
    device = TextConverter(resource_manager, returned_string, codec=codec, laparams=layout_parameters)
    fp = open(pdf_path_string, 'rb')
    interpreter = PDFPageInterpreter(resource_manager, device)
    password = ""
    max_pages = 0
    caching = True
    page_numbers = set()

    for page in PDFPage.get_pages(fp, page_numbers, maxpages=max_pages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = returned_string.getvalue()

    fp.close()
    device.close()
    returned_string.close()
    return text


def smooth_gauss(y, box_pts):
    box = np.ones(box_pts) / box_pts
    mu = int(box_pts / 2.0)
    sigma = 50  # seconds

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu) / sigma) ** 2))

    box = box / np.sum(box)
    sum_value = 0
    for ind in range(0, box_pts):
        sum_value += box[ind] * y[ind]

    return sum_value


def remove_repeats(array):
    array_no_repeats = np.unique(array, axis=0)
    array_no_repeats = array_no_repeats[np.argsort(array_no_repeats[:, 0])]
    return array_no_repeats


def remove_nans(array):
    array = array[~np.isnan(array).any(axis=1)]
    array = array[~np.isinf(array).any(axis=1)]
    return array

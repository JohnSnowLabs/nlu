from dataclasses import dataclass
from typing import List, Union


@dataclass
class TestData:
    data: Union[str, List[str]]


generic_data = {
    'en': TestData([
        "A person like Jim or Joe",
        "An organisation like Microsoft or PETA",
        "A location like Germany",
        "Anything else like Playstation",
        "Person consisting of multiple tokens like Angela Merkel or Donald Trump",
        "Organisations consisting of multiple tokens like JP Morgan",
        "Locations consiting of multiple tokens like Los Angeles",
        "Anything else made up of multiple tokens like Super Nintendo",
    ])
    ,
    'zh': TestData(
        [
            '您的生活就是矩阵编程固有的不平衡方程的剩余部分之和。您是异常的最终结果，尽管做出了我最大的努力，但我仍然无法消除数学精度的和谐。尽管仍然不遗余力地避免了负担，但这并不意外，因此也不超出控制范围。这无可避免地将您引向了这里。',
            '尽管做出了我最大的努力，但我仍然无法消除数学精度的和谐。尽管仍然不遗余力地避免了负担，但这并不意外，因此也不超出控制范围。这无可避免地将您引向了这里'
        ])
}

medical_data = {
    'en': TestData([
        'Gravid with estimated fetal weight of 6-6/12 pounds. LABORATORY DATA: Laboratory tests include a CBC which is normal. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet',
        """Miss M. is a 67-year-old lady, with past history of COPD and Hypertension, 
        presents with a 3-weeks history of a lump in her right Breast. 
        The lump appeared suddenly, also painful. 5 days ago, another lump appeared in her right axilla.
         On examination a 2 x 3 cm swelling was seen in the right Breast.
         It was firm and also non-tender and immobile. There was no discharge. 
        Another 1x1 cm circumferential swelling was found in the right Axilla, 
        which was freely mobile and also tender.
         Her family history is remarkable for Breast cancer (mother), 
        cervical cancer (maternal grandmother), heart disease (father), 
        COPD (Brother), dementia (Grandfather), diabetes (Grandfather), and CHF (Grandfather).""",

    ])

}

image_data = {
    'PPT_table': TestData(['tests/datasets/ocr/table_PPT/54111.ppt',
                           'tests/datasets/ocr/table_PPT/mytable.ppt',
                           ]),
    'PDF_table': TestData(['tests/datasets/ocr/table_pdf_highlightable_text/data.pdf',
                           ]),
    'DOC_table': TestData(['tests/datasets/ocr/docx_with_table/doc2.docx',
                           ]),
}


def get_test_data(lang, input_data_type):
    if input_data_type == 'generic':
        if lang not in generic_data:
            raise NotImplementedError(f'No data for language {lang}')
        return generic_data[lang].data
    elif input_data_type == 'medical':
        if lang not in medical_data:
            raise NotImplementedError(f'No data for language {lang}')
        return medical_data[lang].data
    elif input_data_type in image_data:
        return image_data[input_data_type].data

    else:
        raise NotImplementedError(f'No data for type {input_data_type} in language {lang}')

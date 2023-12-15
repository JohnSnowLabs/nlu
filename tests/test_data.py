from dataclasses import dataclass
from typing import List, Union
import pandas as pd


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
        "Disney Comics was a comic book publishing company operated by The Walt Disney Company which ran from 1990 to 1993.",
        "I really liked that movie!",
        "Peter love pancaces. I hate Mondays",
        "Donald Trump from America and Angela Merkel from Germany dont share many opinions.",
        "You stupid person with an identity that shall remain unnamed, such a filthy identity that you have go to a bad place you person!",
        "<!DOCTYPE html> <html> <head> <title>Example</title> </head> <body> <p>This is an example of a simple HTML page with one paragraph.</p> </body> </html>",
        "HELLO WORLD! How are YOU!?!@",
        "I liek penut buttr and jelly",
        'John told Mary he would like to borrow a book',

    ])
    ,
    'de': TestData(
        [
            "Wer ist Praesident von Deutschland",
            "Was ist NLP?",
        ])
    ,
    'zh': TestData(
        [
            '您的生活就是矩阵编程固有的不平衡方程的剩余部分之和。您是异常的最终结果，尽管做出了我最大的努力，但我仍然无法消除数学精度的和谐。尽管仍然不遗余力地避免了负担，但这并不意外，因此也不超出控制范围。这无可避免地将您引向了这里。',
            '尽管做出了我最大的努力，但我仍然无法消除数学精度的和谐。尽管仍然不遗余力地避免了负担，但这并不意外，因此也不超出控制范围。这无可避免地将您引向了这里'
        ])
    ,
    'tr': TestData(
        [
            "Dolar yükselmeye devam ediyor.",
            "Senaryo çok saçmaydı, beğendim diyemem.",
        ])
    ,
    'fr': TestData(
        [
            "NLU est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python.",
            "A aller voir d'urgence !",
        ])
    ,
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
        "The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also given 1 unit of Metformin daily. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day.",
        "DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin",
        "interferon alfa-2b 10 million unit ( 1 ml ) injec",
        "The patient has cancer and high fever and will die next week.",
        "The patient has COVID. He got very sick with it.",
        "Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer.",
        "Covid 19 is",
        "The doctor pescribed Majezik for my severe headache.",
        "The patient is a 71-year-old female patient of Dr. X. and she was given Aklis, Dermovate, Aacidexam and Paracetamol.",


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
    'IMG_vit': TestData(['tests/datasets/ocr/vit/general_images/images/bluetick.jpg',
                         'tests/datasets/ocr/vit/general_images/images/chihuahua.jpg',
                         'tests/datasets/ocr/vit/general_images/images/egyptian_cat.jpeg',
                         'tests/datasets/ocr/vit/ox.jpg',
                         'tests/datasets/ocr/vit/general_images/images/hen.JPEG',
                         'tests/datasets/ocr/vit/general_images/images/hippopotamus.JPEG',
                         'tests/datasets/ocr/vit/general_images/images/junco.JPEG',
                         'tests/datasets/ocr/vit/general_images/images/palace.JPEG',
                         'tests/datasets/ocr/vit/general_images/images/tractor.JPEG'
                         ]),
    'IMG_classifier': TestData(['tests/datasets/ocr/images/teapot.jpg']),
}

audio_data = {
    'asr': TestData(['tests/datasets/audio/asr/ngm_12484_01067234848.wav']),
}

qa_data = {
    'tapas': TestData([(pd.DataFrame({'name':['Donald Trump','Elon Musk'], 'money': ['$100,000,000','$20,000,000,000,000'], 'age' : ['75','55'] }),[
            "Who earns less than 200,000,000?",
            "Who earns 100,000,000?",
            "How much money has Donald Trump?",
            "How old are they?",
        ])]
        ),
    'qa': TestData(['What is my name?|||My name is CKL']),
}

summarizer_data = {
    'summarizer': TestData(['''LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.'''
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

    elif input_data_type in audio_data:
        return audio_data[input_data_type].data

    elif input_data_type in summarizer_data:
        return summarizer_data[input_data_type].data

    elif input_data_type in qa_data:
        if input_data_type in ["tapas"]:
            return qa_data[input_data_type].data[0]
        else:
            return qa_data[input_data_type].data
    else:
        raise NotImplementedError(f'No data for type {input_data_type} in language {lang}')

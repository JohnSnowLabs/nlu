


import unittest
from nlu import *
class TestT5(unittest.TestCase):

    def test_t5(self):
        pipe = nlu.load('en.t5.small',verbose=True)
        data = [
            'The first matrix I designed was quite naturally perfect. It was a work of art. Flawless. Sublime. A triumph only equaled by its monumental failure.',
            'I love peanut butter and jelly',
            'Who is president of America',
            'Who is president of Germany',
            'What is your favorite  food'
        ]
        pipe['t5'].setTask('translate English to French')
        df = pipe.predict(data, output_level='document')
        print ("French:")
        print(df['T5'])
        print(df.columns)

        pipe['t5'].setTask('translate English to German')
        df = pipe.predict(data, output_level='document')
        print ("German:")
        print(df['T5'])
        print(df.columns)


        pipe['t5'].setTask('Question')
        df = pipe.predict(data, output_level='document')
        print ("Question:")
        print(df['T5'])
        print(df.columns)


        pipe['t5'].setTask('Make it sad')
        df = pipe.predict(data, output_level='document')
        print ("SAD:")
        print(df['T5'])
        print(df.columns)



        pipe['t5'].setTask('Make it stupid')
        df = pipe.predict(data, output_level='document')
        print ("STUPID:")
        print(df['T5'])
        print(df.columns)



        pipe['t5'].setTask('Make it angry')
        df = pipe.predict(data, output_level='document')
        print ("ANGRY:")
        print(df['T5'])
        print(df.columns)



        pipe['t5'].setTask('Translate English to German')
        df = pipe.predict(data, output_level='document')
        print ("GERMAN:")
        print(df['T5'])
        print(df.columns)



        pipe['t5'].setTask('cola sentence:')
        df = pipe.predict(data, output_level='document')
        print ("COLA:")
        print(df['T5'])
        print(df.columns)


        pipe['t5'].setTask('translate English to Spanish')
        df = pipe.predict(data, output_level='document')
        print ("Spanish:")
        print(df['T5'])
        print(df.columns)


    def test_task1_cola(self):
        pipe = nlu.load('en.t5.base',verbose=True)
        data = [
            'John made Bill master of himself',
            'Anna and Mike is going skiing and they is liked is',
            'Anna and Mike like to dance'
        ]


        pipe['t5'].setTask('cola sentence:')
        res = pipe.predict(data)
        print('TEST Task 1 : Sentence acceptability judgment,CoLA')
        print(res['T5'])

    def test_task2_RTE(self):
        pipe = nlu.load('en.t5.base',verbose=True)
        data = [
            'Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years.  sentence2: Johnny is a millionare.',
            'Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years.  sentence2: Johnny is a poor man.',

            'It was raining in England for the last 4 weeks. sentence2: Yesterday, England was very wet. ',
            'I live in italy. sentence2: I live in Europe',
            'Peter loves New York, it is his favorite city. sentence2: Peter loves new York.'
        ]


        pipe['t5'].setTask('rte sentence1:')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 2 : Natural language inference,RTE')
        print(res.columns)
        print(res[['T5','document']])


    def test_task3_MNLI(self):
        pipe = nlu.load('en.t5.base',verbose=True)
        data = [
            'At 8:34, the Boston Center controller received a third, transmission from American 11.    premise: The Boston Center controller got a third transmission from American 11.'

        ]


        pipe['t5'].setTask('mnli hypothesis:')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 3 : Natural language inference, MNLI')
        print(res.columns)
        print(res[['T5','document']])



    def test_task4_MRPC(self):
        pipe = nlu.load('en.t5.base',verbose=True)
        data = [
            'We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said . sentence2: Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11"',
            'We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said . Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11"'

            ' It is raining hot dogs!  I like ice cream',
            ' It was 40 degrees all day. It was pretty hot today',

            ' It is raining hot dogs!      sentence2: I like ice cream',
            ' It was 40 degrees all day', '      sentence2: It was pretty hot today',


        ]


        pipe['t5'].setTask('mrpc sentence1:')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 4 : Natural language inference, MNLI')
        print(res.columns)
        print(res[['T5','document']])


    def test_task5_QNLI(self):
        pipe = nlu.load('en.t5.base',verbose=True)
        data = [
          'Where did Jebe die?     sentence: Ghenkis Khan recalled Subtai back to Mongolia soon afterwards, and Jebe died on the road back to Samarkand ',
          'What does Steve like to eat?    sentence: Steve watches TV all day'


        ]


        pipe['t5'].setTask('qnli question:')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 5 : Natural language inference, QNLI')
        print(res.columns)
        print(res[['T5','document']])


    def test_task6_QQP(self):
        data = [
            'What attributes would have made you highly desirable in ancient Rome?        question2: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?',
            'What was it like in Ancient rome?      question2: What was Ancient rome like?'

                ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('qqp question1:')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 6 : Natural language inference, QQP')
        print(res.columns)
        print(res[['T5','document']])



    def test_task7_SST2(self):
        data = [
            'it confirms fincher ’s status as a film maker who artfully bends technical know-how to the service of psychological insight',
            'I hated that movie'

        ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('sst2 sentence: ')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 7 : BINARY SENTIMENT, SST2')
        print(res.columns)
        print(res[['T5','document']])



    def test_task8_STSB(self):
        data = [
            'What attributes would have made you highly desirable in ancient Rome?        sentence2: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?',
            'What was it like in Ancient rome?      sentence2: What was live like as a King in Ancient Rome?',

        ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('stsb sentence1: ')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 8 : Regressive Sentence Similarity , STSB')
        print(res.columns)
        print(res[['T5','document']])


    def test_task9_CB(self):
        data = [
            'Valence was helping       premise: Valence the void-brain, Valence the virtuous valet. Why couldn’t the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping',
            'What attributes would have made you highly desirable in ancient Rome?        premise: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?',
            'What was it like in Ancient rome?      premise: What was live like as a King in Ancient Rome?',
            'Peter lov'

        ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('cb hypothesis: ')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 9 : CB ')
        print(res.columns)
        print(res[['T5','document']])



    def test_task10_COPA(self):
    
        data = [
            'Many citizens relocated to the capitol. choice2: Many citizens took refuge in other territories. premise: Political violence broke out in the nation. question: effect',
            '  He fell off the ladder.    choice2:    He climbed up the lader       premise: The man lost his balance on the ladder. question:  what happend was result?',
            '  He fell off the ladder.    choice2:    He climbed up the lader       premise: The man lost his balance on the ladder. question:  effect',
            '  He fell off the ladder.    choice2:    He climbed up the lader       premise: The man lost his balance on the ladder. question:  correct',
            '  many citizens relocated to the capitol.   choice2:    Many citizens took refuge in other territories        premise :  Politcal Violence broke out in the nation.      question: effect',
            '  many citizens relocated to the capitol.   choice2:    Many citizens took refuge in other territories        premise :  Politcal Violence broke out in the nation.      question: correct',
            '  many citizens relocated to the capitol.   choice2:    Many citizens took refuge in other territories        premise :  Politcal Violence broke out in the nation.      question: bananas?',

            '  The assailant struck the man in the head.     choice2:    The assailant took the man’s wallet.        premise:   The man fell unconscious.   question: What was the cause if this?',
            '  The assailant struck the man in the head.     choice2:    The assailant took the man’s wallet.        premise:   The man fell unconscious.   question: effect',
            '  The assailant struck the man in the head.     choice2:    The assailant took the man’s wallet.        premise:   The man fell unconscious.   question: correct',
            '  The assailant struck the man in the head.     choice2:    The assailant took the man’s wallet.        premise:   The man fell unconscious.',
            ' He was in the kitchen cooking          choice2: He was at work         choice3: He went to the mooon     choice4: He went to the gym and worked out       premise :   The man ate a peanut butter sandwich' ,
            ' He went tasdasdasdo the gym and worked dasdaout    choice2: He was at work         choice3: He went to the mooon     choice4: He was in the kitchen cooking       premise :   The man ate a peanut butter sandwich',
            ' He went to theasdasdas gdasym dasand dwasdsorked out    choice2: He was at work         choice3: He went to the mooon     choice4: He was in the kitchen cooking       premise :   The man ate a peanut butter sandwich     question: correct'

        ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('copa choice1: ')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 10 : COPA ')
        print(res.columns)
        print(res[['T5','document']])



    def test_task11_MultiRc(self):
        
        data = [
            # paragraph:

        '''
         Why was Joey surprised the morning he woke up for breakfast?
         answer:       There was a T-REX in his garden.
         paragraph:
         Sent 1:       Once upon a time, there was a squirrel named Joey.          
         Sent 2:       Joey loved to go outside and play with his cousin Jimmy.          
         Sent 3:       Joey and Jimmy played silly games together, and were always laughing.          
         Sent 4:       One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond.          
         Sent 5:       Joey woke up early in the morning to eat some food before they left.          
         Sent 6:       He couldn’t find anything to eat except for pie!          
         Sent 7:       Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast.          
         Sent 8:       After he ate, he and Jimmy went to the pond.          
         Sent 9:       On their way there they saw their friend Jack Rabbit.          
         Sent 10:      They dove into the water and swam for several hours.          
         Sent 11:      The sun was out, but the breeze was cold.          
         Sent 12:      Joey and Jimmy got out of the water and started walking home.          
         Sent 13:      Their fur was wet, and the breeze chilled them.          
         Sent 14:      When they got home, they dried off, and Jimmy put on his favorite purple shirt.          
         Sent 15:      Joey put on a blue shirt with red and green dots.          
         Sent 16:      The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.
         ''',


        '''
         Why was Joey surprised the morning he woke up for breakfast?
         answer:       There was only pie to eat.
         paragraph:
         Sent 1:       Once upon a time, there was a squirrel named Joey.          
         Sent 2:       Joey loved to go outside and play with his cousin Jimmy.          
         Sent 3:       Joey and Jimmy played silly games together, and were always laughing.          
         Sent 4:       One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond.          
         Sent 5:       Joey woke up early in the morning to eat some food before they left.          
         Sent 6:       He couldn’t find anything to eat except for pie!          
         Sent 7:       Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast.          
         Sent 8:       After he ate, he and Jimmy went to the pond.          
         Sent 9:       On their way there they saw their friend Jack Rabbit.          
         Sent 10:      They dove into the water and swam for several hours.          
         Sent 11:      The sun was out, but the breeze was cold.          
         Sent 12:      Joey and Jimmy got out of the water and started walking home.          
         Sent 13:      Their fur was wet, and the breeze chilled them.          
         Sent 14:      When they got home, they dried off, and Jimmy put on his favorite purple shirt.          
         Sent 15:      Joey put on a blue shirt with red and green dots.          
         Sent 16:      The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.''',


        ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('multirc question: ')
        res = pipe.predict(data, output_level='document')
        print('TEST Task 11 : MultiRC - Question Answering ')
        print(res.columns)
        print(res[['T5','document']])

    def test_task12_WiC(self):
        data = [
            '''
             sentence1:    The airplane crash killed his family.     
             sentence2:    He totally killed that rock show!. 
             word :        kill 
             ''',

            '''
             sentence1:    The expanded window will give us time to catch the thieves.     
             sentence2:    You have a two-hour window of turning in your homework. 
             word :        window 
             ''',

            '''
             sentence1:    He jumped out the window.     
             sentence2:    You have a two-hour window of turning in your homework. 
             word :        window 
             '''

            ,

        ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('wic pos: ')#wic pos:
        res = pipe.predict(data, output_level='document')
        print('TEST Task 12 : WiC - Word sense disambiguation ')
        print(res.columns)
        print(res[['T5','document']])



    def test_task13_WSC_DPR(self):
        # todo
        data = [
            'wsc:     The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made *it* pleasant and airy.',
           'wsc :          The party was really crazy and  a lot of people had fun until *it* ended.',
            'wsc :          The party was really crazy but the the car killed everybody,  *it* was going so fast!.',

        ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('')#wsc:
        res = pipe.predict(data, output_level='document')
        print('TEST Task 13 : WSC - Coreference resolution  ')
        print(res.columns)
        print(res[['T5','document']])



    def test_task14_text_summarization(self):

        data = [
            'the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth .'

        ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('summarize: ')#wsc:
        res = pipe.predict(data, output_level='document')
        print('TEST Task 14 : Summarization  ')
        print(res.columns)
        print(res[['T5','document']])


    def test_task15_SQuAD_question_answering(self):

        data = [
            'What does increased oxygen concentrations in the patient’s lungs displace? context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.',
            'What did Joey eat for breakfast ?    context : Once upon a time, there was a squirrel named Joey. Joey loved to go outside and play with his cousin Jimmy. Joey and Jimmy played silly games together, and were always laughing. One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond. Joey woke up early in the morning to eat some food before they left. for breakfast. After he ate, he and Jimmy went to the pond. On their way there they saw their friend Jack Rabbit. They dove into the water and swam for several hours. The sun was out, but the breeze was cold. Joey and Jimmy got out of the water and started walking home. Their fur was wet, and the breeze chilled them. When they got home, they dried off, and Jimmy put on his favorite purple shirt. Joey put on a blue shirt with red and green dots. The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed,'
        ]

        pipe = nlu.load('en.t5.base',verbose=True)
        pipe['t5'].setTask('question: ')#wsc:
        res = pipe.predict(data, output_level='document')
        print('TEST Task 15 : SQuAD question answering  ')
        print(res.columns)
        print(res[['T5','document']])


    def test_pre_config_t5_summarize(self):
        data = [
            'the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth .'
        ]

        pipe = nlu.load('en.t5.summarize',verbose=True)

        res = pipe.predict(data)
        print(res.columns)
        print(res[['T5','document']])

    def test_pre_config_t5_summarize_alias(self):
        data = [
            'the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth .'
        ]

        pipe = nlu.load('summarize',verbose=True)

        res = pipe.predict(data)
        print(res.columns)
        print(res[['T5','document']])
        pipe.print_info()
if __name__ == '__main__':
    unittest.main()








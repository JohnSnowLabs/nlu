import unittest
import nlu

class TestSeqMPNetClassifier(unittest.TestCase):

    def test_mpnet_sequence_classifier(self):
        # Load the specific NLU pipeline for sequence classification
        pipe = nlu.load("en.classify.mpnet._ukr_message")
        
        # New data points to classify
        data = [
            "I love driving my car.",
            "The next bus will arrive in 20 minutes.",
            "Pineapple on pizza is the worst 🤮"
        ]
        
        # Predict the classification for each data point
        df = pipe.predict(data, output_level="document")
        
        # Print each column of the dataframe to inspect the prediction results
        for c in df.columns:
            print(df[c])

if __name__ == "__main__":
    unittest.main()




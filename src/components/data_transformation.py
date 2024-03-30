import os
import re
from dataclasses import dataclass
from src.logger.logging import logging
from src.utils.utils import save_object
import nltk
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.exception.exception import CustomException


@dataclass
class PreprocessorConfig:
    Preprocessor_file_path=os.path.join('artifacts','GRU_Preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = PreprocessorConfig()

    def clean_text(self, text):
        try:
            logging.info("Text Cleaning is starting")
            # Defining variables
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            
            # Cleaning Process
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
            text = text.lower() 
            text = text.split()
            text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
            cleaned_text = " ".join(text)
            logging.info("Text Cleaning is ended")
            return cleaned_text
        except Exception as e:
            logging.exception("An error occurred during text cleaning")
            raise CustomException(e)

    def transform(self, df):
        try:
            # Split input and output in X and y
            X = df['Review']
            y = df['Rating'] - 1

            # Clean the text data
            X_cleaned = [self.clean_text(text) for text in X]
            
            # Define the Tokenizer
            GRU_tokenizer = Tokenizer(num_words = 44778)

            # Fitting the data to tokenizer
            GRU_tokenizer.fit_on_texts(X_cleaned)

            # Tokenize the cleaned text data
            X_sequences = GRU_tokenizer.texts_to_sequences(X_cleaned)
            
            # Pad the tokenized sequences
            X_padded = pad_sequences(X_sequences, maxlen=1889, padding = 'pre')
            
            # Resampling the imbalanced labels
            ros = RandomOverSampler(random_state=0)
            X_balanced, y_balanced = ros.fit_resample(X_padded, y)

            logging.info("Data transformation pipeline is completed")
            
            preprocess = Preprocessor(GRU_tokenizer)
            save_object(
                file_path=self.data_transformation_config.Preprocessor_file_path,
                obj=preprocess
            )
                      
            return X_balanced, y_balanced

        except Exception as e:
            logging.exception("An error occurred during data transformation pipeline")
            raise CustomException(e)
        

class Preprocessor:
    def __init__(self,tokenizer):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = tokenizer
        
    def clean_text(self, text):
        # Remove non-alphabetic characters (including spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
        text = text.lower() 
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        cleaned_text = " ".join(tokens)
        return cleaned_text
        
    def transform(self, X):
        X_cleaned = [self.clean_text(text) for text in X]
        X_sequences = self.tokenizer.texts_to_sequences(X_cleaned)
        X_padded = pad_sequences(X_sequences, maxlen=1889, padding='pre')
        return X_padded
    




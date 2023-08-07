import step3_inference
from config import FinetunePath


class Interface(object):
    def __init__(self):
        self.database = []
        self.model = None
        
    def load_database(self, database_filepath):
        '''
        Loads a database from a filepath
        '''
        try:
            with open(database_filepath, encoding = 'utf8') as file:
                self.database = file.readlines()
            print("Database Loaded!")
        except:
            print("Please load a valid database filepath")
            
    def load_model(self, model_filepath = FinetunePath, mode = 's'):
        try:
            self.model = step3_inference.Inference(model_filepath,mode)
        except:
            print("Please load a valid model filepath")
            
    
    def database_topk(self, character, k=0):
        '''
        Searches a database using list comprehensions for similar characters to the
        character provided. K provides nearest neighbors, and will give all matches
        by default, but can be specified to give fewer similairities. 
        '''
        if len(self.database) == 0:
            print("Please load a valid database filepath")
        else:
            similar = [line.strip() for line in self.database if character in line]
            similar = set("".join(similar))
            similar.discard(" ")
            similar = list(similar)
            if len(similar) == 0:
                return ""
            if k == 0:
                return similar
            return similar[:k]
    
    def model_topk(self, character, k=5):
        '''
        Uses the trained model in step3_inference in order to generate similar 
        characters to the character provided. K again specifies how many similar
        matches are desired. 
        '''
        if self.model is None:
            print("Please load a valid model filepath")
        else:
            return self.model.get_topk(character, k)[0][0]
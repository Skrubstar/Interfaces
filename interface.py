import step3_inference
from config import FinetunePath
import re
import os
import json
import numpy as np
import random
import math
import time
import openai
import torchmetrics
import sklearn.metrics
import backoff
import requests
import pandas as pd
import sklearn

class Interface(object):
    def __init__(self):
        self.database = []
        self.model = None
        self.current_storage = []
        self.all_words = set([])
        self.banned = '，、。 ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghijklmnopqrstuvwxyz(){},./\><?"|+-1234567890*%$#@!^&[]？！）（\n；“：”Ａ—:MⅠT×↓;Ｔα°Ｅ２Ｃｍ'
        
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
        
    def all_words_create(self, filepath):
        '''
        Creates the list of every single character from a given file.
        This is used as the basis to categorize what is a Chinese character
        for the randomization without similarity. 
        '''
        str_file=""
        with open(filepath, "r+", encoding="utf8") as f:
            str_file = f.read()
        for char in str_file:
            if char not in self.banned:
                self.all_words.add(char)
        self.all_words = list(self.all_words)
    
    def count_diff(self, x, y):
        '''
        Counts the difference between two inputs,
        going elementwise and returning the total count 
        and the indecies where the differences occur. 
        '''
        count = 0
        ctr = 0
        difference = []
        for a, b in zip(x, y):
            if a != b:
                count+=1
                difference.append(ctr)
            ctr+=1
        return [count, difference]
    
    def json_file_open(self, filepath):
        '''
        Opens the contents for a json file.
        '''
        contents = open(filepath, encoding = 'utf8')
        return json.load(contents)
    

    def file_open(self, filepath):
        '''
        Opens the contents for a file
        '''
        file = []
        with open(filepath, "r+", encoding='utf8') as f:
            file = f.readlines()
        return file

    def file_process(self, open_filepath, storage_filepath):
        '''
        Takes a file in the format of the CBLUE-2 Dataset and 
        reads it, adding newline characters if needed. Finally,
        writes the opriginal sentence and the randomized sentence
        in the form:
        Original Sentence
        Randomized Sentence
        '''
        json_file = self.json_file_open(open_filepath)
        og_file = []
        randomized_file = []
        for item in json_file:
            item1 = item["text"]
            if item1[-1] != "\n":
                item1+="\n"
            og_file.append(item1)
            randomized_file.append(self.rand_sentence(item1))
        with open(storage_filepath, "a+", encoding="utf8") as file:
            for i in range(len(og_file)):
                file.write(og_file[i])
                file.write(randomized_file[i])


    def get_completion(self, prompt , model ="gpt-3.5-turbo", temp = 0.2, p = 0.1):
        '''
        Used to call ChatGPT using OpenAI's API. Temperature is randomness, and top_p
        is the selection from what portion of that randomness.
        '''
        messages = [{"role": "user", "content": prompt }]
        response = openai.ChatCompletion.create(
                model = model ,
                messages = messages,
                temperature = temp,
                top_p = p,
                n=10)
        return response.choices [0].message
    
    def correct_file(self, string_storage ,start = 0, end = -1):
        '''
        Takes a list of strings, and calls upon ChatGPT to correct them.
        The start/end and the reason why it stores to a class list is the fact
        that ChatGPT often times out. The start/end and class variable allows for 
        an easy way to call the method again without having to reprocess the 
        whole dataset. 
        '''
        if end==-1:
            end=len(string_storage)
        print("Total",len(string_storage[start:end]))
        ctr = start
        for string in string_storage[start:end]:
            prompt = f"""
            You will be provided with one sentence in Chinese surrounded by triple quotes. 
        
            Step 1 - Scan each sentence for errors. Each provided sentence will be of the correct character
            length, and requires no addition or deletion in the sentence. 
            Step 2 - Correct each sentence for errors while preserving the original sentence length.
            Each sentence will only have characters that require correction, rather than any errors in spacing and 
            puncuation or any other error that will alter the length of the sentence. Preserve the exact character 
            length, even if it means leaving an error uncorrected. If a sentence has no errors, return the exact 
            original sentence.
            Step 3 - Return the one corrected sentences in the same format and the same length
            as the provided input text, and verify that the length is exactly the same between the input
            and the corrected sentence.
        
            Answer only in this format:
            C: - | | -
        
            \"\"\"{string}\"\"\"
            """
            print(ctr)
            response = self.get_completion(prompt)
            self.current_storage.append(response["content"])
        return self.current_storage

    def rand_sentence(self, sentence, long_length = 10, long_replace = 7, short_replace=3):
        '''
        Randomizes a sentence. First by calculating how many characters to replace in a long
        sentence, and a short sentence, as well as calculating what is counted as a long sentence.

        Then, replaces a character by randomly getting an index for a character and checking to see
        if it is a valid character to be replaced, IE determined to be a Chinese character via not 
        being in banned.

        If the character is in the database, it will look for a similar character to replace.
        If not, it will draw from the whole pool of words found for the replacement.
        From there, it ensures we don't have a duplicate in that dataset, and then returns
        the sentence with the new randomization.

        It also keeps track of the used indices to avoid randomizing or retrying to randomize
        the same index multiple times. Note that this may lead to high run times in cases of
        the repalcement count being similar to the length of the sentence, as the randomization
        process needs to be run multiple times to find a valid index to randomize upon. 
        '''
        replace = long_replace
        if len(sentence) < long_length:
            replace = short_replace
        all_words_copy = self.all_words.copy()
        used = []
        replaced = 0
        sent = sentence
        while (replaced < replace and len(used)<len(sentence)):
            idx = random.randint(0, len(sentence)-2)
            while idx in used:
                idx = random.randint(0, len(sentence)-1)
            char = sentence[idx]
            if (char not in self.banned):
                sim = self.database_topk(char)
                if (len(sim)> 1):
                    rand_idx = random.randint(0, len(sim)-1)
                    randed = sim[rand_idx]
                    while randed== char:
                        sim.pop(rand_idx)
                        rand_idx = random.randint(0, len(sim)-1)
                        randed = sim[rand_idx]
                    sent = sent[0:idx]+randed+sent[idx+1:]
                else:
                    rand_idx = random.randint(0, len(all_words_copy)-1)
                    randed = all_words_copy[rand_idx]
                    while randed== char:
                        all_words_copy.pop(rand_idx)
                        rand_idx = random.randint(0, len(all_words_copy)-1)
                        randed = all_words_copy[rand_idx]
                    sent = sent[0:idx]+randed+sent[idx+1:]
                replaced += 1
            used.append(idx)
        return sent
    
            
    def banned_expand(self, characters):
        '''
        Adds characters to the banned list
        '''
        for char in characters:
            self.banned+=str(char)
        
    def get_current_storage(self):
        return self.current_storage
    
    def get_all_words(self):
        return self.all_words
    
    def get_database(self):
        return self.database
    
    def get_banned(self):
        return self.banned
#use genetic algorithm to generate a pickup lines

import random
import string
import time
import sys
import os
import math

#target = "To be or not to be"
target = "I love you"

#make a list of possible characters
possible_characters = string.ascii_letters + string.punctuation + string.digits + " "

#make a function to generate a string of the same length as the target

def generate_string(length):
    return ''.join(random.choice(possible_characters) for i in range(length))


#make simple fitness function
def fitness_function(guess):
    score = 0
    for i in range(len(guess)):
        if guess[i] == target[i]:
            score += 1
    return score

#make simulation function
while True:
    

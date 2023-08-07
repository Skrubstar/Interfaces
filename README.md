# Interfaces
This interface that can either use list comprehensions to search a character database, 
and give all similar characters found on the same line, and another method in order to 
use the model from char_sim.py in pytorch-soft-masked-bert in order to return a 
list of the nearest similarities found.

step3_inference.py has also been slightly modified to accept alternative File Paths for the 
model, and to return the k closest characters instead of only the top 5. 

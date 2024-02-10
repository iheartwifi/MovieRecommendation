#import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#CSV file path
file = 'C:\\Users\\Admin\\OneDrive\\Pictures\\Documents\\GitHub\\MovieRecommendation\\movies.csv'

#open and read CSV file 
movies_df = pd.read_csv(file)

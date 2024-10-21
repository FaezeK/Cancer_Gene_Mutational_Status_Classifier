##############################################################################################################
## This is a script to test the performance of random forest on the balanced set of selected tumour types 
## that showed the best performance for each gene of interest.
##############################################################################################################

# import required libraries
import pandas as pd
import timeit
import sys
from sklearn.ensemble import RandomForestClassifier

# variables provided at run-time
gene_of_interest = sys.argv[1]

# timing the run-time
start_time = timeit.default_timer()

######################################
########## Read input files ##########
######################################
print('Reading input files ...')
print('')
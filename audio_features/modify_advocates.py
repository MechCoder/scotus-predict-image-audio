import pandas as pd
import re

def convert_string(x):
    return ''.join([t.lower() for t in re.split('[, . ]', x)])

advocates = pd.read_csv("data/input/advocates.csv")
advocates["lawyer_name"] = advocates["lawyer_name"].apply(convert_string)
advocates.to_csv("data/input/modifiedAdvocates.csv")

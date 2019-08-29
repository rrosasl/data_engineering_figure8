
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def etl_pipeline(messages='messages.csv', categories='categories.csv'):

    import pandas as pd
    '''
    INPUT:
    > messages: type(csv) tweets from disaster relieve
    > categories: type(csv) categories of interest for disaster relieve e.g. related-1;request-0;offer-0;aid_related-0;medi...

    PROCESS:
    >  Import libraries and load datasets
    >  Merge datasets
    >  split categories into separate columns
    >  Convert category values to just numbers 0 or 1.
    >  Replace categories column in df with new category columns.
    >  Remove duplicates
    >  Save the clean dataset into an sqlite database

    OUTPUT:
    
    > Returns sql data

    '''
    #Read data
    messages = pd.read_csv(messages)
    categories = pd.read_csv(categories)

    # Merge both data sets
    df = pd.merge(messages, categories, on ='id', how='inner')

    # Change from 1 column of categories to multiple columns with binary
    categories = pd.DataFrame(df.categories.str.split(';',expand=True)) #expand columns
    category_colnames = categories.iloc[0,:].str[:-2] # get column names
    categories.columns = category_colnames # change column names

    #format
    last_char = lambda x : x[-1]
    for cat in categories:
        categories[cat] = categories[cat].apply(last_char) #delete last character
        categories[cat] = categories[cat].astype(int)  #change to integer

    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')

    # drop duplicates
    df.drop_duplicates(subset='id',keep='first',inplace=True)



    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql('messages_labeled', engine, index=False)

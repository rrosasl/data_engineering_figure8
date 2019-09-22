import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    Reads the messages and categories and returns them on a pandas DataFrame
    
    Args:
        messages_filepath (str): location of messages 
        categories_filepath(str): location of categories
        
    Returns:
        Pandas DataFrame
        
    '''
    #Read data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge both data sets
    df = pd.merge(messages, categories, on ='id', how='inner')
    
    return df


def clean_data(df):
    '''
    Cleans the data Frame 
    Args: 
        df (pandas DataFrame): loaded dataframe
        
    returns: 
        clean dataFrame
        
    '''
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
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('labeled_messages', engine, index=False, if_exists='replace')
    engine.dispose()  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    
#print(sys.argv[1:])
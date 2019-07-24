import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on = 'id', right_on = 'id', how = 'inner')
    return df


def clean_data(df):
    categories = df['categories'].str.split(';', expand = True)
    names = categories.loc[0,]
    names = pd.Series([name[:-2] for name in names])
    categories.rename(columns = names, inplace = True)
    for column in categories:
    # set each value to be the last character of the string
    #categories[column] = [int(categories.loc[i,column].split('-')[1]) for i in range(0,categories.shape[0])]

        categories[column] = categories[column].str[-1]    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop(columns = 'categories', inplace = True)
    df = pd.concat([df,categories], axis = 1)
    df.drop_duplicates(inplace = True)
    return df
    


def save_data(df, database_filename):
    database_filename = 'sqlite:///' + database_filename
    engine = create_engine(database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    pass  


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
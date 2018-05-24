
# coding: utf-8

# ### Library Functions

# In[1]:


import re
import json
import spacy
import math
import os.path
import sys, os, unicodedata
import pandas as pd

from collections import defaultdict
from datetime import datetime, timedelta
from pandas.io.json import json_normalize


# In[2]:


# File Path Constants
URL_IGNORE_LIST_PATH              = './configuration/URL Ignore List.json'
KEYWORD_IGNORE_LIST_PATH          = './configuration/Keyword Ignore List.txt'
USER_SEGMENT_LIST_PATH            = './configuration/part-00000-2aa20d63-3e3a-47d2-8bed-d199cef5b814-c000.json'
DATERANGE_CONFIGURATION_PATH      = './configuration/daterange.txt'
AMAZON_WEB_SERVICE_E3_BASE_FOLDER = r'C:\Users\liferay\Documents\analytics data\export'


# In[3]:


MINIMUM_TOPIC_OF_INTEREST_THRESHOLD_SCORE = 1
DECAY_MULTIPLIER_BASE = .90

# https://stackoverflow.com/questions/11066400/remove-punctuation-from-unicode-formatted-strings/11066687#11066687
PUNCTUATION_UNICODE_TABLE = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
NON_ENGLISH_URL_REGEX = re.compile('\/zh(_CN)?\/'
                                   '|\/fr(_FR)?\/'
                                   '|\/de(_DE)?\/'
                                   '|\/it(_IT)?\/'
                                   '|\/ja(_JP|-JP)?\/'
                                   '|\/pt(-br|_BR|_PT)?\/'
                                   '|\/es(-es|_ES)?\/'
                                   '|\/ru\/')
WWW_OR_CUSTOMER_LIFERAY_URL_REGEX = re.compile(r'^https://www\.liferay|^https://customer\.liferay')
BOT_AND_CRAWLER_REGEX = re.compile('((.*)(bot|Bot)(.*)'
                                   '|(.*)spider(.*)'
                                   '|(.*)crawler(.*)'
                                   '|HubSpot'
                                   '|CloudFlare\-AlwaysOnline'
                                   '|WkHTMLtoPDF)')
PARENTHESIS_REGEX = re.compile(u'\(.*?\)')
BANNED_KEYWORDS_LIST = []
INTEREST_CALCULATION_WINDOW_TIMEDELTA = timedelta(30)

DATE_RANGE_OPTIONS = {
    'day'   : timedelta(1),
    'week'  : timedelta(7),
    'month' : timedelta(30)
}

UTM_PARAMETERS = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content']
HUBSPOT_PARAMETERS = ['_hsenc', '_hsmi', '__hstc', '__hssc', '__hsfp']
GOOGLE_ANALYTICS_PARAMETERS = ['_ga', '_gac']
URL_REDIRECT_PARAMETERS = ['redirect', '_3_WAR_osbknowledgebaseportlet_redirect']
ALL_OPTIONAL_URL_PARAMETERS = UTM_PARAMETERS + HUBSPOT_PARAMETERS + GOOGLE_ANALYTICS_PARAMETERS + URL_REDIRECT_PARAMETERS

with open(KEYWORD_IGNORE_LIST_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        BANNED_KEYWORDS_LIST.append(line.strip())

nlp = spacy.load('en')

# Populate URL Ignore List
URL_IGNORE_LIST_MATCH = []
URL_IGNORE_LIST_CONTAINS = []

with open(URL_IGNORE_LIST_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        json_result = json.loads(line)      
        comparison_type = json_result['Type']
        
        if comparison_type == 'match':
            URL_IGNORE_LIST_MATCH = json_result['URLs']
        elif comparison_type == 'contains':
            URL_IGNORE_LIST_CONTAINS = json_result['URLs']
        else:
            print("UNEXPECTED TYPE: {}".format(comparison_type))

START_DATE_STRING = 0
END_DATE_STRING = 0
START_DATE_DATETIME = 0
END_DATE_DATETIME = 0
CALCULATE_YESTERDAY_ONLY = False


# In[4]:


# Read configuration file for start/end dates
with open(DATERANGE_CONFIGURATION_PATH, 'r', encoding='utf-8') as f:
    
    # First parameter is to calculate 'all' or only 'yesterday' topics of interest
    for line in f:
        # Ignore lines starting with a pound-sign
        if line.startswith('#'):
            continue
        else:
            if line.strip() == 'yesterday':
                CALCULATE_YESTERDAY_ONLY = True
            break
                
    # Second parameter is the start date
    for line in f:
        # Ignore lines starting with a pound-sign
        if line.startswith('#'):
            continue
        else:
            START_DATE_STRING = line.strip()
            START_DATE_DATETIME = datetime.strptime(line.strip(), '%Y%m%d')
            break
            
    # Third parameter is for end date
    for line in f:
        # Ignore lines starting with a pound-sign
        if line.startswith('#'):
            continue
        else:
            if line == 'yesterday':
                END_DATE_DATETIME = (datetime.today() - timedelta(1))
                END_DATE_STRING = END_DATE_DATETIME.strftime('%Y%m%d')
            else:
                END_DATE_STRING = line.strip()
                END_DATE_DATETIME = datetime.strptime(line.strip(), '%Y%m%d')


# In[5]:


if False:
    print(START_DATE_STRING)
    print(END_DATE_STRING)

    print(START_DATE_DATETIME)
    print(END_DATE_DATETIME)

    print(CALCULATE_YESTERDAY_ONLY)


# #### Augment Tokenizer
# The tokenizer fails on many hypenated words, so I wanted to augment it to work better.
# Examples: 
# 
# * State-of-the-art collaboration platform targets quality patient care.
# * Share files with a simple drag-and-drop. Liferay Sync transforms the Liferay platform into a central and secure easy-to-use document sharing service.
# * Importing/Exporting Pages and Content - portal - Knowledge | "Liferay

# In[6]:


import spacy
from spacy.attrs import *

#from spacy.symbols import ORTH, POS, TAG

# Source: https://github.com/explosion/spaCy/issues/396


nlp = spacy.load('en')
nlp.tokenizer.add_special_case(u'state-of-the-art', [{ORTH: 'state-of-the-art',
                                                      LEMMA: 'state-of-the-art', 
                                                      LOWER: 'state-of-the-art',
                                                      SHAPE: 'xxxxxxxxxxxxxxxx',
                                                      POS: 'ADJ', 
                                                      TAG: 'JJ'}])
nlp.tokenizer.add_special_case(u'State-of-the-art', [{ORTH: 'State-of-the-art',
                                                      LEMMA: 'state-of-the-art', 
                                                      LOWER: 'state-of-the-art',
                                                      SHAPE: 'xxxxxxxxxxxxxxxx',
                                                      POS: 'ADJ', 
                                                      TAG: 'JJ'}])
nlp.tokenizer.add_special_case(u'drag-and-drop', [{ORTH: 'drag-and-drop',
                                                      LEMMA: 'drag-and-drop', 
                                                      LOWER: 'drag-and-drop',
                                                      SHAPE: 'xxxxxxxxxxxxx',
                                                      POS: 'ADJ', 
                                                      TAG: 'JJ'}])


# In[7]:


# Library Functions

import re
import langdetect
import string
from collections import OrderedDict
from langdetect.lang_detect_exception import ErrorCode, LangDetectException
from string import printable


def playFinishedSound():
    """
    This is for alerting me that something has finished executing.
    This will play a sound.
    """
    from pygame import mixer
    mixer.init()
    mixer.music.load('./configuration/finished.mp3')
    mixer.music.play()

def replace_punctuation(text):
    """
    The purpose of this function is to replace non-ASCII punctuation with its equivalent.
    """
    return text.replace("’", "'")

def segmentWordsIntoKeyWordPhraseList(words, debug=False):

    phrase_list = []
    
    if debug: print("\nOriginal Sentence: {}".format(words))
    # First segment the words by '|' or '-'
    split_words = re.split(r'[\|]| \- ', words)
    split_words = [s.strip() for s in split_words]
    cleaned_up_and_split_sentences = []
    
    # Search for instances of acronymns surrounded in parenthesis. Ex: (DXP)
    # Remove those, and add it automatically to the phrase list
    for sentence in split_words:
        terms_within_parenthesis = [term[1:-1] for term in re.findall(PARENTHESIS_REGEX, sentence)]
        phrase_list += terms_within_parenthesis
        if debug: print(terms_within_parenthesis)
            
        remaining_text = ''.join(re.split(PARENTHESIS_REGEX, sentence))
        cleaned_up_and_split_sentences.append(remaining_text)
        if debug: print(remaining_text)
        
    for sentence in cleaned_up_and_split_sentences:
        if debug: print("Sentence: {}".format(sentence))
        doc = nlp(sentence)
        for chunk in doc.noun_chunks:
            if debug: print("\tText: {} \n\tRoot: {} \n\tRoot Dependency: {} \n\tRoot Head: {}".format(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text))
            
            text = chunk.text
            if debug:
                print(text)
                print("\tPOS: {}".format(chunk[0].pos_))
                print("\tTag: {}".format(chunk[0].tag_))
                print("\tChunk[0]: {}".format(chunk[0]))
                
            # Skip keywords that contain CD (Cardinal Numbers) for now
            if 'CD' in [c.tag_ for c in chunk]:
                print("Skipping, contains CD")
                continue
            
            # Skip URLs
            url_found = False
            for token in chunk:
                if debug: print(token)
                if token.like_url:
                    url_found = True
                    print("Skipping, URL Detected! ({})".format(text))
                    break
                    
            if url_found:
                continue
            
            # We'll skip a phrase for now if it contains a number
            # E.g. Free download: Gartner's evaluation of 21 digital 
            # experience platform (DXP) providers based on their completeness of vision and ability to execute
            
            # CD - [5 Critical Things] Nobody Tells You About Building a Journey Map
            # Recursively remove until no more? - These six customer experience trends will shape business in 2018
            if chunk[0].tag_ in ['DT', 'PRP$', 'WP', 'PRP', 'WRB', 'CD', ':']:
                if debug: print("Starting 'ignore word' found in: {}".format(chunk))
                #text = ' '.join(s.text for s in chunk[1:])
                
                unwanted_text = chunk[0].text
                if debug: print("Unwanted text: {}".format(unwanted_text))
                text = chunk[1:].text
                
                # If we shrunk it down to nothing
                if not text:
                    continue
            
            # Removes invisible characters
            printable_string = ''.join(char for char in text.strip() if char in printable)
            
            # Converts string to lower case; if matches criteria
            # Note: Keep acroynmns the same, check if 2 or more letters, and all caps
            printable_string = modifyCapitalizationOfWords(printable_string)
            
            #if 'blog' in printable_string:
            #    print("Original Sentence: [{}]".format(words))
            #    print("Blog Word: [{}]".format(printable_string))
            
            if text == chunk.root.text:
                phrase_list.append(printable_string)
            else:
                phrase_list.append(printable_string)
                #phrase_list.append(chunk.root.text.lower())
            
    if debug: print("Final list: {}".format(phrase_list))
    return phrase_list
    
def modifyCapitalizationOfWords(text):
    """
    This function will take the given noun phrase, and adjust captialization as necessary.
    Currently it only retains acronymn capitalization.
    I should ventually add a proper noun list as well.
    """
    
    updated_text = [word if (len(word) >=2) and (word.upper() == word) else word.lower() for word in text.split()]
    
    return ' '.join(updated_text)
    
def isEnglish(text, debug=False):
    
    # Empty String
    if not text.strip():
        return False
    
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        if debug:
            print("Failed Unicode Detector")
        return False

    try:
        possible_language_list = langdetect.detect_langs(text)
        
        if debug:
            print(possible_language_list)
        
        for entry in possible_language_list:
            if ((entry.lang == 'en') and (entry.prob > .50)):
                return True
    
        return False

    except LangDetectException:
        print("**** Language Exception caught!")
        display("Original Text: [{}]".format(text))
    
    return True


# In[8]:


def get_list_of_date_folders(start_date='20180227', end_date='20180326'):
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    step = timedelta(days=1)

    list_of_date_folder_names = []

    while start_date <= end_date:
        date_string = start_date.date().strftime('%Y%m%d')
        list_of_date_folder_names.append(date_string)
        start_date += step

    return list_of_date_folder_names

def read_json_as_list(full_file_path):
    all_web_browsing_history = []

    with open(full_file_path, 'r', encoding='utf-8') as f:
        for counter, line in enumerate(f):
            dict_entry = json.loads(line)       
            all_web_browsing_history.append(dict_entry)
                
    return all_web_browsing_history
                

def convert_string_of_json_to_df(list_of_json):
    start_time = datetime.now()
    df = json_normalize(list_of_json)
    print("\tExecution Time: {}".format(datetime.now() - start_time))
    return df


# ### Populate Segment Info

# In[9]:


# Populate Segment Information
segment_lookup_df = pd.DataFrame()
json_list = read_json_as_list(USER_SEGMENT_LIST_PATH)
segment_lookup_df = json_normalize(json_list)
display(segment_lookup_df)
segment_lookup_df = segment_lookup_df.set_index(['identifier', 'datasource', 'datasourceindividualpk'])['segmentnames'].apply(pd.Series).stack()
segment_lookup_df = pd.DataFrame(segment_lookup_df)
segment_lookup_df = segment_lookup_df.reset_index().rename(columns={0 : 'segmentName'})

# Switch order of columns
segment_lookup_df = segment_lookup_df[['segmentName', 'identifier', 'datasource', 'datasourceindividualpk']]


if False:
    display(temp_df)
    for index, row in temp_df.groupby('segmentName'):
        print("index")
        display(index)
        print("row")
        display(row)
        print("identifier")
        display(row['identifier'])
        break
        


# In[10]:


if False: display(segment_lookup_df)


# ### ETL Functions

# In[11]:


import re
import numpy as np
import pandas as pd
from furl import furl

def show_dataframe_length_before_and_after(f, df):
    print("\tBefore: {}".format(len(df)))
    df = f(df)
    print("\tAfter: {}".format(len(df)))
    return df

def keep_only_unload_events(df):
    df = df[df['eventid'] == 'unload']
    return df

def remove_all_bots(df):
    df = df[~df['context.crawler'].str.contains('True', na=False)]
    df = df[~df['context.userAgent'].str.match(BOT_AND_CRAWLER_REGEX, na=False)]
    return df

def remove_non_english_urls(df):
    df = df[~df['context.url'].str.contains(NON_ENGLISH_URL_REGEX, na=False)]
    return df

def populate_url_ignore_list(df):
    
    URL_IGNORE_LIST_MATCH_REGEX_STRING    = '|'.join(['^{}$'.format(s.strip()) for s in URL_IGNORE_LIST_MATCH])
    URL_IGNORE_LIST_CONTAINS_REGEX_STRING = '|'.join(URL_IGNORE_LIST_CONTAINS)

    # TODO: Maybe use 'normalized_url' only?
    df['Ignore URL'] = df['context.url'].str.match(URL_IGNORE_LIST_MATCH_REGEX_STRING)                      | df['context.og:url'].str.match(URL_IGNORE_LIST_MATCH_REGEX_STRING)                      | df['context.url'].str.match(URL_IGNORE_LIST_CONTAINS_REGEX_STRING)                      | df['context.og:url'].str.match(URL_IGNORE_LIST_CONTAINS_REGEX_STRING)
    return df

def remove_non_customer_www_lr_urls(df):
    df = df[df['context.url'].str.contains(WWW_OR_CUSTOMER_LIFERAY_URL_REGEX, na=False)]
    return df

def remove_empty_user_id_entries(df):
    df['userid'].replace('', np.nan, inplace=True)
    df.dropna(subset=['userid'], inplace=True)
    return df

def __removeUrlParameters(url, parameter_list):  
    f = furl(url)
    remaining_parameters = { k: f.args[k] for k in f.args if k not in parameter_list }
    f.args = remaining_parameters    
    return f.url

def populateNormalizedUrlField(df):
    df['normalized_url'] = df['context.og:url'].fillna(df['context.url'])
    df['normalized_url'] = df['normalized_url'].apply(lambda x: __removeUrlParameters(x, ALL_OPTIONAL_URL_PARAMETERS))
    return df

def replaceBlankSpacesWithNan(df):
    # '\s+' is 1 or more
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    return df

def filterUnwantedColumns(df):
    wanted_columns_list = ['eventdate', 
                           'analyticskey', 
                           'userid', 
                           'eventid', 
                           'Ignore URL',
                           'normalized_url',
                           'context.url', 'context.og:url', 
                           'context.title', 'context.og:title', 
                           'context.description', 'context.og:description', 
                           'context.keywords', 
                           'eventproperties.scrollDepth', 
                           'eventproperties.viewDuration', 
                           'context.userAgent', 
                           'context.platformName', 
                           'context.browserName', 
                           'context.country', 
                           'context.region', 
                           'context.city', 
                           'clientip']
    df = df[wanted_columns_list]
    return df

def convertColumnsToAppropriateDataTypes(df):
    print("Converting eventdate to datetime objects")
    df['eventdate'] = pd.to_datetime(df['eventdate'])
    return df


# ### Read JSON files and save as DataFrame

# In[12]:


get_ipython().run_cell_magic('time', '', '# Plan go through list of directories, and parse in all the relevant JSON files.\n\nstart_date = START_DATE_STRING\nend_date = END_DATE_STRING\n\nlist_of_date_folder_names = get_list_of_date_folders(start_date=start_date, end_date=end_date)\nfull_df = pd.DataFrame()\n\nstart_time = datetime.now()\n\nfor sub_folder_name in list_of_date_folder_names:\n    directory_name = os.path.join(AMAZON_WEB_SERVICE_E3_BASE_FOLDER, sub_folder_name)\n    \n    for filename in os.listdir(directory_name):\n        full_directory_and_file_name = os.path.join(directory_name, filename)\n        \n        if filename.endswith(".json"): \n            try:\n                print("\\n{}".format(full_directory_and_file_name))\n                json_list = read_json_as_list(full_directory_and_file_name)\n                print("\\tEntries: {}".format(len(json_list)))\n                df = convert_string_of_json_to_df(json_list)\n\n                # XXX: Workaround to improve memory usage\n                df = keep_only_unload_events(df)\n\n                full_df = full_df.append(df, ignore_index=True)\n            except:\n                print("Unexpected error detected!") ')


# In[13]:


get_ipython().run_cell_magic('time', '', '\nimport warnings\n\n# Surpress Warning Messages from "removing non-English URLs"\nwarnings.filterwarnings("ignore", \'This pattern has match groups\')\n\nprint("Keeping only UNLOAD events")\netl_df = show_dataframe_length_before_and_after(keep_only_unload_events, full_df)\n\nprint("Removing Bots")\netl_df = show_dataframe_length_before_and_after(remove_all_bots, etl_df)\n\nprint("Removing Non-English URLs")\netl_df = show_dataframe_length_before_and_after(remove_non_english_urls, etl_df)\n\nprint("Removing non-customer, non-www URLs")\netl_df = show_dataframe_length_before_and_after(remove_non_customer_www_lr_urls, etl_df)\n\nprint("Removing empty userid entries")\netl_df = show_dataframe_length_before_and_after(remove_empty_user_id_entries, etl_df)\n\nprint("Populating normalized_url field")\netl_df = populateNormalizedUrlField(etl_df)\n\nprint("Populating URL Ignore List")\netl_df = show_dataframe_length_before_and_after(populate_url_ignore_list, etl_df)\nprint("Ignoring {} URLs".format(len(etl_df[etl_df[\'Ignore URL\'] == True])))\n\nprint("Removing unwanted columns")\netl_df = filterUnwantedColumns(etl_df)\n\nprint("Converting columns to appropriate data types")\netl_df = convertColumnsToAppropriateDataTypes(etl_df)\n\nprint("Replacing blank spaces with NaN")\netl_df = replaceBlankSpacesWithNan(etl_df)')


# In[14]:


get_ipython().run_cell_magic('time', '', '\n# Make a copy, and use it\nclean_df = etl_df.copy()\ndisplay("Length: {}".format(len(clean_df)))')


# ### Save URLs for Web Scraping

# In[15]:


# Disable for production (for now)
if False:
    url_s = pd.Series(clean_df['normalized_url'].unique()).sort_values()
    print("Number of URLs: {}".format(len(url_s)))
    url_s.to_csv('./output/Unique Visitor URLs.csv', index=False)


# ### Create DataFrame: URL Lookup Information
# This will be the centralized URL to information Data Frame.

# In[16]:


get_ipython().run_cell_magic('time', '', "\n\nurl_to_title          = clean_df.groupby(['normalized_url'])['context.title'].apply(set)\nurl_to_og_title       = clean_df.groupby(['normalized_url'])['context.og:title'].apply(set)\nurl_to_description    = clean_df.groupby(['normalized_url'])['context.description'].apply(set)\nurl_to_og_description = clean_df.groupby(['normalized_url'])['context.og:description'].apply(set)\nurl_to_keywords       = clean_df.groupby(['normalized_url'])['context.keywords'].apply(set)")


# In[17]:


def createUrlToKeywordDf():
    columns = ['normalized_url',
           'analyticsclient.merged_title', 
           'analyticsclient.merged_description', 
           'analyticsclient.merged_keywords',
           'analyticsclient.generated_keywords']

    url_to_keyword_df = pd.DataFrame(columns=columns)
    url_to_keyword_df['normalized_url'] = clean_df['normalized_url'].unique()
    #display(url_to_keyword_df)
    return url_to_keyword_df

def generateKeywordsFromTitleDescriptionKeywords(title, og_title, description, og_description, keywords, debug=False):
    merged_title = title.union(og_title)
    merged_description = description.union(og_description)
    
    keywords_from_title = set()
    keywords_from_description = set()
    keywords_from_keywords = set()
    
    only_english_titles = set()
    only_english_descriptions = set()
    only_english_keyword_set = set()
    
    title_description_to_keyword_cache = defaultdict(int)

    for entry in merged_title:

        # Skip empty strings       
        if pd.isnull(entry):
            continue
            
        # remove weird HTML punct
        entry = replace_punctuation(entry)
        
        cached_result = title_description_to_keyword_cache[entry]
        
        if cached_result != 0:
            keywords_from_title.update(cached_result)
            only_english_titles.update([entry])
        elif isEnglish(entry):
            #print("isEnglish() passed")
            #print("entry: ", entry)
            keyword_phrase_list = segmentWordsIntoKeyWordPhraseList(entry, debug=False)
            keywords_from_title.update(keyword_phrase_list)
            only_english_titles.update([entry])
            #print("entry: {}".format(entry))
            #print("only_english_titles: {}".format(only_english_titles))
            
            # Update Cache:
            title_description_to_keyword_cache[entry] = keyword_phrase_list
        else:
            print("Non-English detected: [{}]".format(entry))
            title_description_to_keyword_cache[entry] = []
            continue
    
    for entry in merged_description:        
        # Skip empty strings
        if pd.isnull(entry):
            continue
            
        # remove punct
        entry = replace_punctuation(entry)
        
        cached_result = title_description_to_keyword_cache[entry]
        
        if cached_result != 0:
            keywords_from_description.update(cached_result)
            only_english_descriptions.update([entry])
        elif isEnglish(entry):
            keyword_phrase_list = segmentWordsIntoKeyWordPhraseList(entry)
            keywords_from_description.update(keyword_phrase_list)
            only_english_descriptions.update([entry])
            
            # Update Cache:
            title_description_to_keyword_cache[entry] = keyword_phrase_list
        else:
            print("Non-English detected: [{}]".format(entry))
            title_description_to_keyword_cache[entry] = []
            continue
        
    for entry in keywords:
        
        # Skip empty strings
        if pd.isnull(entry):
            continue
            
        if isEnglish(entry):
            split_list = [s.strip() for s in entry.split(',')]
            keywords_from_keywords.update(set(split_list if split_list else []))
            only_english_keyword_set.update(set(split_list if split_list else []))
        else:
            print("Non-English detected: [{}]".format(entry))
            continue
    
    # Debugging
    if debug:
        print("\n\tMerged Title: {} => {}".format(only_english_titles, keywords_from_title))
        print("\tMerged Descr: {} => {}".format(only_english_descriptions, keywords_from_description))
        print("\tKeywords:     {} => {}".format(only_english_keyword_set, keywords_from_keywords))
        
    # merge all sets together
    all_keywords_merged = keywords_from_keywords.union(keywords_from_title, keywords_from_description)
    if debug: print("\tAll Keywords: {}".format(all_keywords_merged))

    # We return the English list of inputs we processed, and the final keyword output
    return list(only_english_titles), list(only_english_descriptions), list(only_english_keyword_set), list(all_keywords_merged)

def populateUrlToKeywordDf(url_to_keyword_df, debug=False):
    unique_url_list = url_to_keyword_df['normalized_url'].unique()

    for counter, url in enumerate(unique_url_list):       
        title = url_to_title.get(url, set())
        og_title = url_to_og_title.get(url, set())
        description = url_to_description.get(url, set())
        og_description = url_to_og_description.get(url, set())
        keywords_set = url_to_keywords.get(url, set())

        if debug: 
            print('\n{} / {}'.format(counter, len(unique_url_list)))
            print('{}'.format(url))
        merged_title_list, merged_description_list, merged_keyword_list, generated_keyword_list = generateKeywordsFromTitleDescriptionKeywords(title, og_title, description, og_description, keywords_set)

        # Populate url_to_keyword_df, with keywords
        index = url_to_keyword_df.loc[url_to_keyword_df['normalized_url'] == url]
        if len(index.index.values) > 1:
            print("ERROR: There shouldn't be more than 1 entry for the URL list!")
            print("index: {}".format(index))
            print("index.index.values: {}".format(index.index.values))
            break

        if len(merged_title_list) > 0: 
            url_to_keyword_df.at[index.index.values[0], 'analyticsclient.merged_title'] = merged_title_list

        if len(merged_description_list) > 0: 
            url_to_keyword_df.at[index.index.values[0], 'analyticsclient.merged_description'] = merged_description_list

        if len(merged_keyword_list) > 0: 
            url_to_keyword_df.at[index.index.values[0], 'analyticsclient.merged_keywords'] = merged_keyword_list

        url_to_keyword_df.at[index.index.values[0], 'analyticsclient.generated_keywords'] = generated_keyword_list
        
        if counter % 100 == 0:
            print("{} / {}".format(counter, len(unique_url_list)))
        
    return url_to_keyword_df

def addKeywordBoosting(df, debug=True):
    www_lr_manual_keywords = pd.read_csv('./manually generated keywords/www-lr-manual-keywords.csv')
    customer_lr_manual_keywords = pd.read_csv('./manually generated keywords/customer-lr-manual-keywords.csv')
    all_lr_manual_keywords = www_lr_manual_keywords.append(customer_lr_manual_keywords, ignore_index=True)
    all_lr_manual_keywords = all_lr_manual_keywords[['URL', 'Keywords']]
    all_lr_manual_keywords = all_lr_manual_keywords.dropna(how='any')
    all_lr_manual_keywords['Keywords'] = all_lr_manual_keywords['Keywords'].apply(lambda x: 
                                               [modifyCapitalizationOfWords(s.strip()) for s in x.split(',') if s.strip()])

    # Populate existing url-to-keyword lookup dataframe
    temp_df = pd.merge(df, all_lr_manual_keywords, how='left', left_on='normalized_url', right_on='URL')

    # Rename the "Keywords" column to "manual.keywords"
    temp_df.rename(columns={'Keywords' : 'manual.keywords'}, inplace=True)

    # Rearrange order of columns
    temp_df = temp_df[['normalized_url',
                       'analyticsclient.generated_keywords',
                       'manual.keywords',
                       'analyticsclient.merged_title',
                       'analyticsclient.merged_description', 
                       'analyticsclient.merged_keywords']]

    # Replace analyticsclient.generated_keywords [] with NaN
    temp_df.loc[temp_df['analyticsclient.generated_keywords'].str.len() == 0, 'analyticsclient.generated_keywords'] = np.nan

    # Filter out URLs where the "Automatically Generated Keywords" or "Manually generated Keywords" are missing
    if debug:
        print("Removing entries where both auto & manually generated keywords are missing")
        print("Before: {}".format(len(temp_df)))
    
    with pd.option_context('display.max_rows', 200, 'display.max_columns', None, 'display.max_colwidth', 50):
        display(temp_df)

    temp_df = temp_df[(~temp_df['analyticsclient.generated_keywords'].isnull()) | (~temp_df['manual.keywords'].isnull())]

    #with pd.option_context('display.max_rows', 200, 'display.max_columns', None, 'display.max_colwidth', 50):
    #    display(temp_df)

    if debug:
        print("After: {}".format(len(temp_df)))

    return temp_df

def generateUrlToKeywordDict(df, keyword_types=[''], use_banned_word_list=True, debug=True):
    """
    TODO:
    There will be multiple options for what type of keywords you can select from
    * manual - these are the tags manually added (there aren't that many of these)
    * title_description_keyword - these are the tags provided by the metadata
    * web_scraping - these are the tags generated by web scraping
    """
    import numpy
    
    # Add new empty column to df, for storing the combined keywords
    df['combined keywords'] = np.nan
    df['combined keywords'] = df['combined keywords'].astype(object)
    
    url_s = df['normalized_url'].unique()
    url_lookup_cache = dict()
    no_keywords_urls = []
    
    for counter, url in enumerate(url_s):
        
        if debug: 
            print("\n{} / {} - {}".format(counter, len(url_s), url))
        generated_keyword_list          = df.loc[df['normalized_url'] == url]['analyticsclient.generated_keywords'].values.tolist()
        manually_populated_keyword_list = df.loc[df['normalized_url'] == url]['manual.keywords'].values.tolist()

        # Filter [nan] scenarios
        if numpy.nan in generated_keyword_list:
            generated_keyword_list = []
        elif len(generated_keyword_list) >= 1:
            generated_keyword_list = generated_keyword_list[0]

        if numpy.nan in manually_populated_keyword_list:
            manually_populated_keyword_list = []
        elif len(manually_populated_keyword_list) >= 1:
            manually_populated_keyword_list = manually_populated_keyword_list[0]

        aggregate_keyword_list = list(set(generated_keyword_list).union(set(manually_populated_keyword_list)))

        if use_banned_word_list:
            aggregate_keyword_list = [w for w in aggregate_keyword_list if w.lower() not in BANNED_KEYWORDS_LIST]

        # Cache result
        url_lookup_cache[url] = aggregate_keyword_list
        if debug:
            print("\t{}".format(aggregate_keyword_list))
        
        if not aggregate_keyword_list:
            print("\tWarning: [{}] has 0 entries!".format(url))
            no_keywords_urls.append(url)
            
        # Add the entry back to the dataframe     
        index = df.loc[df['normalized_url'] == url]
        df.at[index.index.values[0], 'combined keywords'] = aggregate_keyword_list
            
    return url_lookup_cache, df, no_keywords_urls

# For Debugging
def lookUpKeywordBreakdownBasedOnUrl(url):
    title = url_to_title.get(url, set())
    og_title = url_to_og_title.get(url, set())
    description = url_to_description.get(url, set())
    og_description = url_to_og_description.get(url, set())
    keywords_set = url_to_keywords.get(url, set())
    
    print("Title: {}".format(title))
    print("og_title: {}".format(og_title))
    print("description: {}".format(description))
    print("og_description: {}".format(og_description))
    print("keywords: {}".format(keywords_set))
    
def generateKeywordToIndividualKeywordList(url_to_keyword_df):
    url_to_keyword_df = url_to_keyword_df[['normalized_url', 'combined keywords']]

    # Expand each normalized_url, into its own keyword row
    expanded_keywords_df = url_to_keyword_df['combined keywords'].apply(lambda x: pd.Series(x))

    url_to_unique_keyword_df = pd.DataFrame()

    for index, row in expanded_keywords_df.iterrows():
        row_df = row.dropna().to_frame(name='unique keyword')
        row_df['normalized_url'] = url_to_keyword_df['normalized_url'].loc[index]   
        url_to_unique_keyword_df = url_to_unique_keyword_df.append(row_df, ignore_index=True)

        if index % 500 == 0:
            print("{} / {}".format(index, len(expanded_keywords_df)))
    
    return url_to_unique_keyword_df


# ### Populate URL to Information Dataframe
# 
# 
# I don't know why this is so resource intensive...
# Maybe because of remove punctuation function?

# In[18]:


get_ipython().run_cell_magic('time', '', 'url_to_keyword_df = createUrlToKeywordDf()\nurl_to_keyword_df = populateUrlToKeywordDf(url_to_keyword_df)\nurl_to_keyword_df = addKeywordBoosting(url_to_keyword_df)\nurl_lookup_cache, url_to_keyword_df, urls_without_keywords_list = generateUrlToKeywordDict(url_to_keyword_df)\nurl_to_unique_keyword_df = generateKeywordToIndividualKeywordList(url_to_keyword_df)')


# In[19]:


# Save URLs without any keywords
# This is meant to be a debugging output
with open('./output/URLs with NO keywords.txt', 'w', encoding='utf-8') as w:
    for counter, url in enumerate(sorted(urls_without_keywords_list)):
        w.write("{}) {}\n".format(counter, url))


# In[20]:


from collections import defaultdict, Counter       

def compute_score_with_df(user_visits_df, global_visits_df, start_date, debug=False):
    """
    Description: This will take the sites that a user has visited, and perform TF-IDF calculations
    to obtain an output score. Note that we also factor in global visits as well.
    
    Input: 
    user_visits_df - This is the dataframe corresponding to an individual's activites
    glbal_visits_df - This is the dataframe for all user's activites
    
    Output:
    ranked_interest_df - Ranked interests. Format: Topic of Interest, Score, Corresponding URLs Visited
    user_visits_df - The user df, but added with keywords associated with the link
    
    """
    
    keyword_to_logscore = calculate_inverse_document_frequency(user_visits_df, global_visits_df, debug=False)
    
    columns = ['Topic of Interest', 'Score', 'Corresponding URLs Visited']
    ranked_interest_df = pd.DataFrame(columns=columns)

    # Iterate through all URLs the user has visited
    for index, entry in user_visits_df.iterrows():
        
        url = entry['normalized_url']        
        aggregate_keyword_list = url_lookup_cache.get(url, [])
        
        # Exponential Decay Factor - Calculate multiplier
        event_date = entry['eventdate']
        multiplier = calculateDecayMultiplier(event_date, start_date)
        
        # Iterate through the individual keywords extracted from the URL
        for keyword in aggregate_keyword_list:
            
            if not keyword:
                print("ERROR, EMPTY KEYWORD DETECTED!")
                print("URL: {}".format(url))
                print("aggregate_keyword_list: {}".format(aggregate_keyword_list))

            existing_row = ranked_interest_df[ranked_interest_df['Topic of Interest'] == keyword]

            if existing_row.empty:
                row = ranked_interest_df.shape[0]               
                ranked_interest_df.loc[row] = [keyword, (keyword_to_logscore[keyword] * multiplier), np.NaN]
                ranked_interest_df['Corresponding URLs Visited'] = ranked_interest_df['Corresponding URLs Visited'].astype(object)
                ranked_interest_df.at[row, 'Corresponding URLs Visited'] = [url]
            else:
                                
                index = ranked_interest_df.index[ranked_interest_df['Topic of Interest'] == keyword]
                column = ranked_interest_df.columns.get_loc('Score')
                updated_score = ranked_interest_df.iloc[index, column].values[0] + (keyword_to_logscore[keyword] * multiplier)
                ranked_interest_df.iloc[index, column] = updated_score
                
                column = ranked_interest_df.columns.get_loc('Corresponding URLs Visited')
                updated_urls = ranked_interest_df.iat[index.values[0], column]
                updated_urls.append(url)                
                ranked_interest_df.iat[index.values[0], column] = updated_urls

    # Sort by logscore before returning
    ranked_interest_df['Score'] = pd.to_numeric(ranked_interest_df['Score'])
    ranked_interest_df.sort_values(by=['Score'], ascending=False, inplace=True)
    
    #
    user_visits_df = pd.merge(user_visits_df, url_to_keyword_df, how='left', on='normalized_url', copy=True)
    user_visits_df = user_visits_df.drop(['analyticsclient.generated_keywords',                                           'manual.keywords',                                           'analyticsclient.merged_title',                                           'analyticsclient.merged_description',                                           'analyticsclient.merged_keywords'], axis=1)
        
    return ranked_interest_df, user_visits_df  


# In[21]:


if False:
    temp_df = clean_df.sample(500)
    temp_df = temp_df[~temp_df['Ignore URL']]

    #df1, df2 = calculateTopicsOfInterestOnDfOfUsers(temp_df, clean_df, start_date, debug=False)

    df1, df2 = compute_score_with_df(temp_df, clean_df, start_date, debug=False)

    #df = pd.DataFrame(df)


    playFinishedSound()

    #display(type(df))

    with pd.option_context('display.max_rows', 200, 'display.max_columns', None, 'display.max_colwidth', 500):
        display(df2)
        display(df1)


# In[22]:


# TODO: Future optimiziation, only count the user visited keywords
def calculate_inverse_document_frequency(user_visits_df, global_df, user_weight=1.0, global_weight=2.0, 
                                         debug=False, fast=True, save_results=False):
    import math
    import numpy
    import operator
    from collections import defaultdict
    from math import log
    
    label_document_count = defaultdict(float)
    label_document_idf = dict()
    document_count = (len(user_visits_df) * user_weight) + (len(global_df) * global_weight)
    
    user_keywords = set()

    if debug:
        print("Document Count: {}".format(document_count))
    
    keyword_to_weighted_frequency_per_document = defaultdict(float)
    
    ###############
    # User Counts #
    ###############
    
    # Iterate through URLs and extract keywords
    for index, user_visit_entry in user_visits_df.iterrows():
        
        # Skip "Ignore URL" entries
        if user_visit_entry['Ignore URL']:
            if debug:
                print("Ignoring URL, skipping: {}".format(user_visit_entry['normalized_url'] ))
            continue
        
        normalized_url = user_visit_entry['normalized_url']        
        aggregate_keyword_list = url_lookup_cache.get(normalized_url, [])
        
        # Iterate through list and update weights
        for keyword in aggregate_keyword_list:
            keyword_to_weighted_frequency_per_document[keyword] += user_weight
            user_keywords.add(keyword)
        
        if debug:
            if len(aggregate_keyword_list) == 0:
                print("[WARNING: User Counts] - 0 keywords detected for url: {}".format(normalized_url))
        
    if debug:
        print("User Counts:")
        for entry in sorted(keyword_to_weighted_frequency_per_document.items(), key=operator.itemgetter(1)):
            print("\t{} => {}".format(entry[0], entry[1]))
    
    #################
    # Global Counts #
    #################
    
    keyword_to_logscore = dict()
    
    counter = 1
    
    # XXX: maybe don't need to do unique, assume it's already unique?
    
    # Iterate through URLs and extract keywords
    for normalized_url in global_df['normalized_url'].unique():
        
        # Skip "Ignore URL" entries
        if user_visit_entry['Ignore URL']:
            if debug:
                print("Ignoring URL, skipping: {}".format(user_visit_entry['normalized_url'] ))
            continue
                
        #normalized_url = user_visit_entry['normalized_url']            
        aggregate_keyword_list = url_lookup_cache.get(normalized_url, [])
        if debug: print("Aggregate Keyword List: {}".format(aggregate_keyword_list))
        
        # Iterate through list and update weights
        for keyword in aggregate_keyword_list:
            keyword_to_weighted_frequency_per_document[keyword] += global_weight
            #print('Updating keyword count')
            #print("{} => {}".format(keyword, keyword_to_weighted_frequency_per_document[keyword]))
        
        if debug:
            if len(aggregate_keyword_list) == 0:
                print("[WARNING: Global Counts] - 0 keywords detected for url: {}".format(normalized_url))
        
        counter += 1
    
    if debug:
        print("Global Counts:")
        for entry in sorted(keyword_to_weighted_frequency_per_document.items(), key=operator.itemgetter(1)):
            print("\t{} => {}".format(entry[0], entry[1]))
                    
    # Convert to Inverse-Log-Scores
    if debug:
        print("Calculating Inverse Log Scores")

    for entry in sorted(keyword_to_weighted_frequency_per_document.items(), key=operator.itemgetter(1), reverse=True):
        
        if entry[0] in user_keywords:
            inverse_log_score = math.log((document_count + 1) / (entry[1] + 1))
            if debug:
                print("{} => {} ({})".format(entry[0], inverse_log_score, entry[1]))

            keyword_to_logscore[entry[0]] = inverse_log_score

    # This is if we want to write the results to an output file
    
    if save_results:
        with open('Inverse Document Frequency Results.txt', 'w', encoding='utf-8') as w: 
            sorted_x = sorted(keyword_to_logscore.items(), key=operator.itemgetter(1), reverse=True)
            
            for word, log_score in sorted_x:
                w.write("{:30} : {:>5.4f}\n".format(word, log_score))
        
        
    return keyword_to_logscore


# In[23]:


def calculateTopicsOfInterestOnDfOfUsers(filter_grouped_user_df, global_df, start_date, debug=False):
    """
    Inputs:
        filter_grouped_user_df - This contains all the users who we're trying to calculate the topics of interest for.
                                 This should be pre-filtered by your own specified date range.
        global_df - This contains all user's viewing history, in the same time period as the filter_grouped_user_df
    
    Outputs:
        user_to_topics_of_interest_df - This is the list of (userid, analyticskey) to (Topics of Interest, scores)
        keyword_to_url_df - This is the user input with keyword list attached to it
    
    """
    
    counter = 1
    user_to_results = dict()
    columns = ['User ID', 'Analytics Key', 'Topic of Interest', 'Score', 'Corresponding URLs Visited']
    user_to_topics_of_interest_df = pd.DataFrame(columns=columns)
    all_keywords_to_url_df = pd.DataFrame()

    for userid_and_analytics_key_tuple, group in filter_grouped_user_df.groupby(['userid', 'analyticskey']):

        user_id = userid_and_analytics_key_tuple[0]
        analytics_key = userid_and_analytics_key_tuple[1]
        
        if debug: 
            print("\n{}) User ID: {} Analytics Key: {}".format(counter, user_id, analytics_key)) 
            
        score_df, user_with_keyword_df = compute_score_with_df(group, global_df, start_date)       
        score_df['User ID'] = user_id
        score_df['Analytics Key'] = analytics_key
        score_df = score_df[columns]
        user_to_topics_of_interest_df = user_to_topics_of_interest_df.append(score_df, ignore_index=True)

        if debug:
            display(user_with_keyword_df)
        
        all_keywords_to_url_df = all_keywords_to_url_df.append(user_with_keyword_df, ignore_index=True)
                
        if counter % 500 == 0:
            print('{} / {}'.format(counter, len(filter_grouped_user_df['userid'].unique())))

        counter += 1
    
    return user_to_topics_of_interest_df, all_keywords_to_url_df


# In[24]:


from datetime import timedelta, datetime

def extractDateRange(df, start_date, date_range='day', debug=False):
    """
    Description:
    This takes in a dataframe, and extracts the rows where the eventdate field is within the date range specified.
    Note that the start_date is inclusive, so if you ask for start_date = Jan 1, and range='day', you get all the 
    data from only Jan 1.
    """
        
    end_date = start_date + DATE_RANGE_OPTIONS.get(date_range, date_range)
    
    if debug:
        print("Start Date: {}".format(start_date))
        print("Date Range: {}".format(date_range))
        print("End Date:   {}".format(end_date))
    
    df = df[(df['eventdate'] > start_date) & (df['eventdate'] < end_date)].sort_values(by='eventdate', ascending=True)
    
    if debug:
        print("Earliest Reported Date: {}".format(df.iloc[0]['eventdate']))
        print("Latest Reported Date:   {}".format(df.iloc[-1]['eventdate']))
    
    return df

# Testing code for function above:
#start_date = datetime(2018, 3, 14)
#end_date = datetime(2018, 4, 1)
#date_range = timedelta(30)

#temporary_df = extractDateRange(clean_df, start_date=start_date, date_range='week', debug=True)

#display(temporary_df)


def calculateDecayMultiplier(event_date, start_date, debug=False):
    day_difference = (start_date - event_date).days
    multiplier = DECAY_MULTIPLIER_BASE ** day_difference
    
    if debug:
        print("Start Date:   {}".format(start_date))
        print("Current Date: {}".format(event_date))
        print("Difference:   {}".format(day_difference))
        print("Multiplier:   {}".format(multiplier))
    
    return multiplier


# In[25]:


def calculateInfoForAllIndividualUsers(user_df, global_df, start_date, end_date, time_period='day', debug=False):
    """
    This function will iterate through all the users from user_df, and return all the individual's scores
    """
    
    current_date = start_date
    all_users_to_topic_of_interest_df = pd.DataFrame()
    
    while current_date < end_date:
        print("current_date: {}".format(current_date))
        
        # We want to look 30-days back for calcuations
        date_range_filtered_user_df = extractDateRange(user_df, 
                                                       start_date=(current_date - INTEREST_CALCULATION_WINDOW_TIMEDELTA), 
                                                       date_range=(INTEREST_CALCULATION_WINDOW_TIMEDELTA + timedelta(1)), 
                                                       debug=False)
        date_range_filtered_global_df = extractDateRange(global_df, 
                                                         start_date=(current_date - INTEREST_CALCULATION_WINDOW_TIMEDELTA), 
                                                         date_range=(INTEREST_CALCULATION_WINDOW_TIMEDELTA + timedelta(1)), 
                                                         debug=False)
        
        # do Interest calculations for individuals
        user_to_topics_of_interest_df, user_keyword_subset_df = calculateTopicsOfInterestOnDfOfUsers(date_range_filtered_user_df, date_range_filtered_global_df, (current_date + timedelta(1)))        
        user_to_topics_of_interest_df['currdate'] = current_date

        # append to larger list
        all_users_to_topic_of_interest_df = all_users_to_topic_of_interest_df.append(user_to_topics_of_interest_df, ignore_index=True)

        current_date += timedelta(1)
        
    return all_users_to_topic_of_interest_df

def calculateInfoForAllIndividualUsersSaveToJSON(user_to_toi_and_score, file_location, debug=False):
    """
    
    
    """
    partition_key = datetime.today().strftime('%Y%m%d0000')
    output_file = open(file_location, 'w', encoding='utf-8')

    # Gameplan:
    # - Go through date/userid/analyticskey
    # - Go through each keyword & score
    # - Find all URLs & Counts that correspond to the keyword
    # - Save info as a JSON entry
    
    for curr_date, row in user_to_toi_and_score.groupby(['currdate']):
        
        partition_key = datetime.today().strftime('%Y%m%d0000')
        curr_date_string = curr_date.strftime("%Y-%m-%d")

        print("\ncurrdate: {}".format(curr_date_string))
        
        for userid_and_analytics_key, row2 in row.groupby(['User ID', 'Analytics Key']):
            user_id = userid_and_analytics_key[0]
            analytics_key = userid_and_analytics_key[1]
            
            if debug:
                print("User ID: {}".format(user_id))
                print("Analytics Key: {}".format(analytics_key))
                
            row2 = row2.sort_values(by=['Score'], ascending=False)
            
            user_to_keyword_info_list = []
                
            for toi_score, row3 in row2.groupby(['Topic of Interest', 'Score']):
                topic_of_interest = toi_score[0]
                score = toi_score[1]

                print("Topic of Interest: {}".format(topic_of_interest))
                print("Score: {}".format(score))
                
                # Generates [URL, View Count]
                url_to_view_count_df = row3['Corresponding URLs Visited'].apply(lambda x: pd.Series(x).value_counts()).T.reset_index()
                url_to_view_count_df.rename(columns={url_to_view_count_df.columns[0] : 'url', url_to_view_count_df.columns[1] : 'visitCount'}, inplace=True)
                
                if debug:
                    display(url_to_view_count_df)
                    
                # print(url_to_view_count_df.to_json(orient='records'))
                
                url_to_visit_count_list = []
                
                for index, url_visit_count in url_to_view_count_df.iterrows():
                    url = url_visit_count['url']
                    visit_count = url_visit_count['visitCount']
                    
                    if debug:
                        print("URL: {}".format(url))
                        print("visitCount: {}".format(visit_count))
                        
                    url_to_visit_count_list.append(OrderedDict([('url', url), 
                                                                ('visitCount', visit_count)]))


                user_to_keyword_info_list.append(OrderedDict([('name', topic_of_interest),
                                                              ('score', score), 
                                                              ('pagesVisited', url_to_visit_count_list)]))

            json_text = json.dumps(
                OrderedDict([('analyticsKey', analytics_key), 
                            ('partitionKey', partition_key),
                            ('userid', user_id),
                            ('Current Date', curr_date_string),
                            ('interests', user_to_keyword_info_list)]))

            output_file.write("{}\n".format(json_text))


# In[26]:


def calculateInfoForAllSegmentsSaveToJSON(segment_to_toi_and_score_df, user_to_toi_and_score, score_threshold, file_location):
    
    partition_key = datetime.today().strftime('%Y%m%d0000')
    output_file = open(file_location, 'w', encoding='utf-8')
    
    for curr_date, row in segment_to_toi_and_score_df.groupby(['currdate']):
        curr_date_string = curr_date.strftime("%Y-%m-%d")
        print("\ncurrdate: {}".format(curr_date_string))       
        user_to_toi_and_score_filtered_by_date_df = user_to_toi_and_score[user_to_toi_and_score['currdate'] == curr_date]
        
        for segment_id, row2 in row.groupby(['segmentIdentifier']):
            print("\tsegmentIdentifier: {}".format(segment_id))
            user_to_toi_and_score_filtered_by_date_and_segment_id_df = getSegmentEntriesDf(user_to_toi_and_score_filtered_by_date_df, segment_id)
            #display(user_to_toi_and_score_filtered_by_date_and_segment_id_df)

            # Create veritcal list of [topic of interest, score]
            keyword_to_score_df = row2.drop(labels=['currdate', 'segmentIdentifier'], axis=1, inplace=False).T.reset_index().copy()
            keyword_to_score_df.columns.values[0] = 'Topic of Interest'
            keyword_to_score_df.columns.values[1] = 'Score'

            #with pd.option_context('display.max_rows', 1000, 'display.max_columns', None, 'display.max_colwidth', 2000):
            #    display(keyword_to_score_df)
                
            keyword_to_url_json_string_list = []

            # Iterate through the current date + segment users, to figure out corresponding URLs
            for index, row3 in keyword_to_score_df.iterrows():
                topic_of_interest = row3['Topic of Interest']
                score = row3['Score']

                

                # Skip NaN values
                if math.isnan(score):
                    #print("Skipping...")
                    continue
                    
                print("\t\t{} : {}".format(topic_of_interest, score))

                # Find corresponding users whose individual scores exceed the threshold
                # Need URL, uniqueVisitsCount
                url_to_counts = getUrlAndUniqueVisitsCount(user_to_toi_and_score_filtered_by_date_and_segment_id_df, topic_of_interest, score_threshold)

                #display(url_to_counts)
                url_to_view_count_list = []
    
                for url, view_count in url_to_counts.items():
                    url_to_view_count_entry = {
                        'url'        : url,
                        'visitCount' : view_count
                    }
                
                    url_to_view_count_list.append(url_to_view_count_entry)
                    
                keyword_entry = {
                    'name'         : topic_of_interest, 
                    'score'        : score, 
                    'pagesVisited' : url_to_view_count_list
                }
                
                keyword_to_url_json_string_list.append(keyword_entry)

            json_text = json.dumps({
                'partitionKey'          : partition_key,
                'segmentIdentifier'     : segment_id,
                'currdate'              : curr_date_string, 
                'interests'             : keyword_to_url_json_string_list
                })
            
            output_file.write("{}\n".format(json_text))
    
def getSegmentEntriesDf(df, segmentIdentifier):
    """
    This will only return rows that match the segmentIdentifier
    """
    
    only_segment_id_entries_df = pd.merge(segment_lookup_df, df, how='inner', left_on='datasourceindividualpk', right_on='User ID', sort=True)#.drop('value', 1)
    #display(only_segment_id_entries_df)
    
    return only_segment_id_entries_df
    
    
def getUrlAndUniqueVisitsCount(df, topic_of_interest, minimum_score_threshold, debug=False):
    
    url_to_unique_visits = OrderedDict()
    
    toi_df = df[(df['Topic of Interest'] == topic_of_interest) 
                & (df['Score'] >= minimum_score_threshold)]
    
    expanded_url_list = toi_df.set_index(['User ID'])['Corresponding URLs Visited'].apply(pd.Series).stack()
    expanded_url_list = pd.DataFrame(expanded_url_list).reset_index().drop(labels=['level_1'], axis=1, inplace=False)
    expanded_url_list.rename(columns={0 : 'Corresponding URLs Visited'}, inplace=True)
    # We are only getting unique: (userid, url) pairs
    no_duplicates_df = expanded_url_list.drop_duplicates(subset=['User ID', 'Corresponding URLs Visited'])
    count_url_visits = no_duplicates_df['Corresponding URLs Visited'].value_counts()
    
    if debug:
        display(toi_df)
        display(expanded_url_list)
        display(no_duplicates_df)
        display(count_url_visits)

    for url_count_tuple in count_url_visits.iteritems():
        url = url_count_tuple[0]
        count = url_count_tuple[1]
        url_to_unique_visits[url] = count
    
    return count_url_visits


# In[27]:


def calculateSegmentWithDf(user_to_topic_of_interest_df, MINIMUM_SCORE_THRESHOLD):    
    user_to_toi_filtered_by_minimum_score = user_to_topic_of_interest_df[user_to_topic_of_interest_df['Score'] > MINIMUM_SCORE_THRESHOLD]
    keyword_to_count = user_to_topic_of_interest_df.groupby('Topic of Interest').count()
    keyword_to_count['Logscore'] = keyword_to_count['User ID'].apply(lambda x: math.log1p(x))
    keyword_to_count = keyword_to_count[['Logscore']]

    return keyword_to_count  

def calculateSegmentInfoFromIndividualDf(segment_name, user_to_toi_df, score_threshold, debug=False):
    """
    This will calculate the interest scores, and
    """
    
    user_to_toi_with_date_df = pd.DataFrame()
    
    # Filter by date
    for index, row in user_to_toi_df.groupby('currdate'):
        
        if debug:
            print("currdate: {}".format(index))
            display(row)
            
        segment_to_topic_of_interest_df = calculateSegmentWithDf(row, score_threshold)
        segment_to_topic_of_interest_transposed_df = segment_to_topic_of_interest_df.T
        segment_to_topic_of_interest_transposed_df['currdate'] = index
        user_to_toi_with_date_df = user_to_toi_with_date_df.append(segment_to_topic_of_interest_transposed_df, ignore_index=True)
        
        if debug:
            display(user_to_toi_with_date_df)

    # Move currdate column to front
    currdate_column = user_to_toi_with_date_df['currdate']
    user_to_toi_with_date_df.drop('currdate', axis=1, inplace=True)
    user_to_toi_with_date_df.insert(0, 'currdate', currdate_column)
    
    # Add Segment Name column
    user_to_toi_with_date_df.insert(1, 'segmentIdentifier', segment_name)
    
    return user_to_toi_with_date_df

def calculateAllSegmentInfo(user_to_toi_df, debug=False):
    """
    This function will return a DataFrame of all segments Topic of Interests and Scores
    """
    
    all_segment_info_df = pd.DataFrame()
    
    # Gameplan:
    # - Iterate through the list of segments
    #   * Filter user_to_toi_df so we only get the users from that segment
    # - Calculate segment toi & scores for that segment
    for segmentName, row in segment_lookup_df.groupby('segmentName'):
        display("segmentName: {}".format(segmentName))
        filtered_user_df = pd.merge(row, user_to_toi_df, how='inner', left_on='datasourceindividualpk', right_on='User ID')
        
        if debug:
            display(filtered_user_df)
            
        if filtered_user_df.shape[0] == 0:
            print("[WARNING] - Segment has 0 users! Skipping...")
            continue

        segment_toi_to_score_df = calculateSegmentInfoFromIndividualDf(segmentName, filtered_user_df, MINIMUM_TOPIC_OF_INTEREST_THRESHOLD_SCORE)
        all_segment_info_df = all_segment_info_df.append(segment_toi_to_score_df, ignore_index=True)
    
    
    
    # Move currdate & segmentIdentifier to front
    currdate_column = all_segment_info_df['currdate']
    segment_id_column = all_segment_info_df['segmentIdentifier']
    all_segment_info_df.drop('currdate', axis=1, inplace=True)
    all_segment_info_df.drop('segmentIdentifier', axis=1, inplace=True)
    all_segment_info_df.insert(0, 'currdate', currdate_column)
    all_segment_info_df.insert(1, 'segmentIdentifier', segment_id_column)
    
    return all_segment_info_df


# ## Pipeline with Output saved as JSON files
# 
# Steps:
# * Filter out the group of users you want as a dataframe
# * Pass in date range for calculations
# * Write output files
#  * Individual -> Topic of Interest (individual topics of interest.json)
#  * Entire Segment -> Topic of Interest (segment topics of interest.json)
#  * Segment URLs Contribution -> Topic of Interest (daily URL contribution to topics of interest.json)

# In[28]:


get_ipython().run_cell_magic('time', '', "\nstart_date = START_DATE_DATETIME\nend_date = END_DATE_DATETIME\n\ntemp_df = clean_df[(~clean_df['Ignore URL']) & (clean_df['eventdate'] >= (start_date - timedelta(1)))]")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Calculating all Individual User\'s Info")\nuser_to_toi_and_score = calculateInfoForAllIndividualUsers(temp_df, clean_df, start_date, end_date, \'day\', True)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Saving Individual Info to JSON file")\ncalculateInfoForAllIndividualUsersSaveToJSON(user_to_toi_and_score, \'./output/individual.json\', debug=False)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Calculating all Segment Info")\nall_segment_info_df = calculateAllSegmentInfo(user_to_toi_and_score, debug=False)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Saving Segment Info to JSON file")\ncalculateInfoForAllSegmentsSaveToJSON(all_segment_info_df, \n                                      user_to_toi_and_score, \n                                      MINIMUM_TOPIC_OF_INTEREST_THRESHOLD_SCORE,\n                                     \'./output/segment.json\')\n\nplayFinishedSound()')


import re
import pandas as pd
import nltk
nltk.download('stopwords')
import string
#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
#defining the object for stemming
porter_stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

""" EMAILS & EMPLOYEERECORDS PREPROCESSING"""

# Create a function to get the time of the day
def find_time_of_day(hour):
    if (hour >= 9) and (hour <= 17):
        return "Working hours (9-5)"
    else:
        return "After hours (the rest)"

# create two dataframes with the data needed
df_emails_original = pd.read_csv('csvs/email headers.csv', encoding='cp1252')
df_employees = pd.read_excel('csvs/EmployeeRecords.xlsx')

# Remove whitespace
df_emails_original.To = df_emails_original.To.str.replace(' ', '')
df_emails_original.From = df_emails_original.From.str.replace(' ', '')
df_employees.EmailAddress = df_employees.EmailAddress.str.replace(' ', '')

# Get only emails out of names
df_emails_original['From_Name'] = df_emails_original['From'].apply(lambda x: x.split('@')[0])
df_emails_original['From_Name'] = df_emails_original['From_Name'].apply(lambda x: x.replace("."," "))

# Create a list with every email in each receivers cell
df_emails_original['To_Name'] = df_emails_original['To'].apply(lambda x: x.split(','))

# Create the same list but now name it differently in order to dave later the list with the employment type of the receiver
df_emails_original['CurrentEmploymentType_To_List'] = df_emails_original['To'].apply(lambda x: x.split(','))

# Match each email in the list of To emails with the employment type
for emails in df_emails_original['To_Name']:
    names = []
    employee_to = []
    for email in emails:
        first_name = df_employees.loc[df_employees['EmailAddress'] == email, 'FirstName'].iloc[0]
        last_name = df_employees.loc[df_employees['EmailAddress'] == email, 'LastName'].iloc[0]
        employee_type = df_employees.loc[df_employees['EmailAddress'] == email, 'CurrentEmploymentType'].iloc[0]
        names.append(last_name + ' ' + first_name)
        employee_to.append(employee_type)
    indices = df_emails_original.loc[df_emails_original['To_Name'].isin([emails])].index.values.tolist()
    # If length of emails is the number of employees add value All
    if len(names) == 54:
        df_emails_original.loc[indices, 'To_Name'] = 'All'
    else:
        df_emails_original.loc[indices, 'To_Name'] = [names]

    # Create a list with the receivers employment type
    df_emails_original.loc[indices, 'CurrentEmploymentType_To_List'] = [employee_to]

# Keep only the unique type
df_emails_original['CurrentEmploymentType_To_Unique'] = df_emails_original['CurrentEmploymentType_To_List'].apply(lambda x: pd.unique(x))

# Convert date column to date
df_emails_original['Date'] = pd.to_datetime(df_emails_original.Date)

# Find time of the day
df_emails_original['just_date'] = df_emails_original['Date'].dt.date

# Create a list with the dates
just_dates = df_emails_original['just_date'].drop_duplicates().tolist()

# Get only the day
df_emails_original['day'] = df_emails_original['Date'].dt.day

# Create a column with the time of the day
df_emails_original['Time_of_day'] = df_emails_original['Date'].dt.hour.apply(find_time_of_day)

# df_subject_only = df_emails_original[['From_Name', 'To_Name','Subject','Date']]

# Keep the original data to create the subject table
df_emails = df_emails_original.copy()

# Create list of To emails
df_emails['To'] = df_emails['To'].apply(lambda x: x.split(','))

# Split the To emails into different columns
df_emails = df_emails.explode('To')

# Drop rows that have the same person in From and To
df_emails = df_emails[df_emails['From'] != df_emails['To']]

# Remove empty spaces
df_emails.To = df_emails.To.str.replace(' ', '')
df_emails.From = df_emails.From.str.replace(' ', '')
df_employees.EmailAddress = df_employees.EmailAddress.str.replace(' ', '')

# Add a suffix that indicates if the person is the sender or the receiver
df_employees_from = df_employees.add_suffix('_From')
df_employees_to = df_employees.add_suffix('_To')

# Merge original data with From
merge_original_from = pd.merge(df_emails_original, df_employees_from, how='left', left_on='From',
                               right_on='EmailAddress_From')

# Create a list to check if the receiver is from one department of nor
merge_original_from['CurrentEmploymentType_To_Unlisted'] = merge_original_from['CurrentEmploymentType_To_Unique'].apply(lambda x: x[0] if len(x)==1 else 'Not unique department')

# Merge emails data with From
merge_from = pd.merge(df_emails, df_employees_from, how='left', left_on='From', right_on='EmailAddress_From')

# Merge From with To
final = pd.merge(merge_from, df_employees_to, how='left', left_on='To', right_on='EmailAddress_To')

# Replace no values with No_name indication
final['LastName_From'] = final['LastName_From'].fillna('No_name')
final['LastName_To'] = final['LastName_To'].fillna('No_name')

# Replace no values with No_title indication
final['CurrentEmploymentTitle_From'] = final['CurrentEmploymentTitle_From'].fillna('No_title')
final['CurrentEmploymentTitle_To'] = final['CurrentEmploymentTitle_To'].fillna('No_title')

# Replace no values with No_type indication
final['CurrentEmploymentType_From'] = final['CurrentEmploymentType_From'].fillna('No_type')
final['CurrentEmploymentType_To'] = final['CurrentEmploymentType_To'].fillna('No_type')

# Add a count column with the value 1 in each line. Needed for the pivot
final['count'] = 1

# Count dates in order to create the graph with the number of emails per date
final['dates_count'] = final.groupby('just_date')['just_date'].transform('count')


# Uncomment the below to extract to csv
# final.to_csv('final.csv')
# merge_original_from.to_csv('merge_original_from.csv')

""" ARTICLES PREPROCESSING"""

# Change this to your path
articles_path = r'C:\Users\user\PycharmProjects\VizAnalytics\articles\\'
articles_df = pd.DataFrame(columns=['article_number', 'content'])


# Defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

# Defining the function for tokenization
def tokenization(text):
    tokens = re.split('W+',text)
    return tokens

# Defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

# Defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

# Defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

# Defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text


# Get all the available articles
for i in range(845):
    f = open(articles_path + '{}.txt'.format(str(i)), "r")
    art_number = i
    word_list = f.read().splitlines()
    articles_df.loc[i,'article_number'] = art_number
    articles_df.loc[i,'content'] = word_list
    f.close()

# List of words without empty strings
for i in range(len(articles_df)):
    without_empty_strings = [string for string in articles_df.loc[i,'content'] if string != ""]
    articles_df.loc[i,'content'] = without_empty_strings
    articles_df.loc[i,'content'] = ' '.join(articles_df.loc[i,'content'])

# Extract dates
def extract_dates(text):
    matches = re.findall(r'(\d{4}\/\d{2}\/\d{2})|(\d{1,2}\s(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s\d{4})',text)
    return matches

# Extract dates from articles
articles_df['dates']= articles_df['content'].apply(lambda x: extract_dates(x))

# Storing the puntuation free text
articles_df['content']= articles_df['content'].apply(lambda x:remove_punctuation(x))
print("Punctuantion free content...\n")

# Extract dates from articles
for i in range(len(articles_df)):
    if articles_df['dates'].iloc[i][0][0] != '':
        articles_df['dates'].iloc[i] = articles_df['dates'].iloc[i][0][0]
    else:
        articles_df['dates'].iloc[i] = articles_df['dates'].iloc[i][0][1]

# Convert to lowercase
articles_df['content']= articles_df['content'].apply(lambda x: x.lower())
print("Lowercase article content...\n")

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in nltk.corpus.stopwords.words('english')]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text

# Remove stop words from content
articles_df['content'] = articles_df['content'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

print(articles_df['content'])
###TF-IDF and K-means
# initialize vectorizer

stop = list(stopwords.words('english'))
stop.extend('20 january 2014'.split())
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.5,stop_words=set(stop))

# fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
X = vectorizer.fit_transform(articles_df['content'])

# initialize KMeans with 2 clusters
kmeans2 = KMeans(n_clusters=3, random_state=0)
kmeans2.fit(X)
tsne_clusters = kmeans2.labels_

# Initialize TSNE with 2 components
tsne = TSNE(n_components=2, learning_rate='auto')
# Pass X to the TSNE
tsne_vecs = tsne.fit_transform(X.toarray())
# Save the two dimensions in x0 and x1
x0 = tsne_vecs[:, 0]
x1 = tsne_vecs[:, 1]
# Assign clusters and TSNE vectors to columns in the original dataframe
articles_df['cluster'] = tsne_clusters
articles_df['x0'] = x0
articles_df['x1'] = x1

def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(tsne_clusters).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score

cluster_map = {0: "C0",
               1: "C1",
               2: "C2"}

# Mapping found through get_top_keywords
articles_df['cluster'] = articles_df['cluster'].map(cluster_map)

# Set image size
plt.figure(figsize=(12, 7))

# Set title
plt.title(" TF-IDF + KMeans with TSNE", fontdict={"fontsize": 18})

# Set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})

# Create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=articles_df, x='x0', y='x1', hue='cluster', palette="pastel")
plt.show()

###Sentiment Analysis
from textblob import TextBlob

# Get specific column for articles
articles_df = articles_df.filter(["article_number", "content", "dates", "cluster"])

# Returns subjectivity
def getSubjectivity(text):
    TextBlob(text).sentiment.subjectivity

# Returns polarity
def getPolarity(text):
    TextBlob(text).sentiment.polarity

# Apply subjectivity and polarity
articles_df['Subjectivity'] = articles_df['content'].apply(getSubjectivity)
articles_df['Polarity'] = articles_df['content'].apply(getPolarity)

# Apply subjectivity and polarity in every column
for i in range(len(articles_df['content'])):
    articles_df['Subjectivity'][i] = TextBlob(articles_df['content'][i]).sentiment.subjectivity
    articles_df['Polarity'][i] = TextBlob(articles_df['content'][i]).sentiment.polarity

# Implement sentiment analysis and create a column to save the values of each row
articles_df['Sentiment'] = ''
articles_df.sort_values(by='dates',inplace=True)
for i in range(len(articles_df)):
    if (articles_df['Polarity'].iloc[i] < 0):
        articles_df['Sentiment'].iloc[i] = 'Negative'
    elif (articles_df['Polarity'].iloc[i] > 0):
        articles_df['Sentiment'].iloc[i] = 'Positive'
    else:
        articles_df['Sentiment'].iloc[i] = 'Neutral'
        articles_df['Polarity'].iloc[i] = -1 * articles_df['Polarity'].iloc[i]

from nltk.tokenize import word_tokenize

#applying tokenization to the column
articles_df['content']= articles_df['content'].apply(lambda x: word_tokenize(str(x)))

stopwords = nltk.corpus.stopwords.words('english')

# Remove stopwords from every row
articles_df['content']= articles_df['content'].apply(lambda x:remove_stopwords(x))

# Apply lemmatization in every row
articles_df['content'] = articles_df['content'].apply(lambda x:lemmatizer(x))

# Find word frequency
def word_freq(str_list):
    new_article = []

    # Gives set of unique words
    unique_words = set(str_list)

    for words in unique_words:
        new_article.append((words, str_list.count(words)))
    new_article.sort(key=lambda y: y[1], reverse=True)
    return new_article

# Create a column with word frequency and add there the appropriate value
articles_df['word_freq'] = articles_df['content'].apply(lambda x: word_freq(x))

# Sort the values
articles_df.sort_values(by = 'article_number',inplace=True)

# Article title extraction

articles_df['article_title'] = ''
lines_list = []
articles_df.sort_values(by='article_number', inplace=True)
for i in range(845):
    f = open(articles_path + '{}.txt'.format(str(i)), "r")

    # Using readlines()
    Lines = f.read().splitlines()
    without_empty_strings = [string for string in Lines if string != ""]
    articles_df['article_title'].iloc[i] = without_empty_strings[1]
    f.close()

articles_df = articles_df.filter(["article_number", "content", "dates",
                                  "cluster", "Subjectivity", "Polarity", "Sentiment", "word_freq", "article_title"])

# Uncomment the below to extract to csv
# articles_df.to_csv('articles_final.csv')
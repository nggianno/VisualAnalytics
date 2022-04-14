import nltk
from wordcloud import WordCloud
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
import pandas as pd
import base64
import dash_bootstrap_components as dbc
from io import BytesIO
import plotly.express as px
from dash import Dash, dash_table
import plotly.graph_objs as go
import random
from collections import Counter
import plotly
import dash


### EMAILS PREPROCESSING ###

# Create a function to get the time of the day
def find_time_of_day(hour):
    if (hour > 4) and (hour <= 8):
        return 'Early Morning'
    elif (hour > 8) and (hour <= 12):
        return 'Morning'
    elif (hour > 12) and (hour <= 16):
        return 'Noon'
    elif (hour > 16) and (hour <= 20):
        return 'Evening'
    elif (hour > 20) and (hour <= 23):
        return 'Night'
    elif (hour > 23) and (hour <= 1):
        return 'MidNight'
    elif (x <= 4):
        return 'Late Night'


# Function needed in order to update the values in the table
def getData(df):
    return df.to_dict('rows')

# create two dataframes with the data needed
df_emails_original = pd.read_csv('3.Disappearance at GAStech/data/data/email headers.csv', encoding='cp1252')
df_employees = pd.read_excel('3.Disappearance at GAStech/data/data/EmployeeRecords.xlsx')

df_emails_original.To = df_emails_original.To.str.replace(' ', '')
df_emails_original.From = df_emails_original.From.str.replace(' ', '')
df_employees.EmailAddress = df_employees.EmailAddress.str.replace(' ', '')

# Get only emails out of names
df_emails_original['From_Name'] = df_emails_original['From'].apply(lambda x: x.split('@')[0])
df_emails_original['From_Name'] = df_emails_original['From_Name'].apply(lambda x: x.replace("."," "))

df_emails_original['To_Name'] = df_emails_original['To'].apply(lambda x: x.split(','))
df_emails_original['CurrentEmploymentType_To_List'] = df_emails_original['To'].apply(lambda x: x.split(','))

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
    if len(names) == 54:
        df_emails_original.loc[indices, 'To_Name'] = 'All'
    else:
        df_emails_original.loc[indices, 'To_Name'] = [names]
    df_emails_original.loc[indices, 'CurrentEmploymentType_To_List'] = [employee_to]

# Create employees list
dept = df_employees['CurrentEmploymentType'].unique().tolist()
for i in dept:
    globals()[f"{i}"] = df_employees.loc[df_employees['CurrentEmploymentType'] == i, 'LastName'].tolist()

executives_png = 'exec.png'
IT_png = 'IT.png'
Engg_png = 'Engg.png'
Fac_png = 'fac.png'
Sec_png = 'security.png'
test_1 = base64.b64encode(open(executives_png, 'rb').read()).decode('ascii')
test_2 = base64.b64encode(open(IT_png, 'rb').read()).decode('ascii')
test_3 = base64.b64encode(open(Engg_png, 'rb').read()).decode('ascii')
test_4 = base64.b64encode(open(Fac_png, 'rb').read()).decode('ascii')
test_5 = base64.b64encode(open(Sec_png, 'rb').read()).decode('ascii')

# Convert date column to date
df_emails_original['Date'] = pd.to_datetime(df_emails_original.Date)

# Find time of the day
df_emails_original['just_date'] = df_emails_original['Date'].dt.date

just_dates = df_emails_original['just_date'].drop_duplicates().tolist()

df_emails_original['day'] = df_emails_original['Date'].dt.day

# Create a column with the time of the day
df_emails_original['Time_of_day'] = df_emails_original['Date'].dt.hour.apply(find_time_of_day)

# df_subject_only = df_emails_original[['From_Name', 'To_Name','Subject','Date']]

# Keep the original data to create the subject table
df_emails = df_emails_original.copy()
df_subject = df_emails_original.copy()

# Remove stopwords for subject
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('of')
stopwords.append('hey')

# Remove RE or FW indication
df_subject['Subject'] = df_subject['Subject'].str.replace("RE: ", "")
df_subject['Subject'] = df_subject['Subject'].str.replace("FW: ", "")

# Drop duplicate subjects
df_subject = df_subject.drop_duplicates(subset=['Subject'])

# # Create list with times of day
# tod = list(df_emails['Time_of_day'].unique())
tod = [{'label': 'Early Morning', 'value': 'Early Morning'},
       {'label': 'Morning', 'value': 'Morning'},
       {'label': 'Noon', 'value': 'Noon'},
       {'label': 'Evening', 'value': 'Evening'},
       {'label': 'Night', 'value': 'Night'},
       {'label': 'MidNight', 'value': 'MidNight'},
       {'label': 'Late Night', 'value': 'Late Night'}]

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

# Merge emails data with From
merge_from = pd.merge(df_emails, df_employees_from, how='left', left_on='From', right_on='EmailAddress_From')

# Merge From with To
final = pd.merge(merge_from, df_employees_to, how='left', left_on='To', right_on='EmailAddress_To')

# Replace no values with No_name indication
final['LastName_From'] = final['LastName_From'].fillna('No_name')
final['LastName_To'] = final['LastName_To'].fillna('No_name')

final['CurrentEmploymentTitle_From'] = final['CurrentEmploymentTitle_From'].fillna('No_title')
final['CurrentEmploymentTitle_To'] = final['CurrentEmploymentTitle_To'].fillna('No_title')

final['CurrentEmploymentType_From'] = final['CurrentEmploymentType_From'].fillna('No_type')
final['CurrentEmploymentType_To'] = final['CurrentEmploymentType_To'].fillna('No_type')

# Add a count column with the value 1 in each line. Needed for the pivot
final['count'] = 1

# Count dates in order to create the graph with the number of emails per date
final['dates_count'] = final.groupby('just_date')['just_date'].transform('count')

# Pivot final df to create the heatmap
graphdf_names = final[['CurrentEmploymentType_From', 'CurrentEmploymentType_To', 'count']]
pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From',
                       columns='CurrentEmploymentType_To',
                       aggfunc='count')

# Fill NaN columns with 0
pivot = pivot.fillna(0)

fig = px.imshow(pivot, width=600, height=600, color_continuous_scale='gray_r')
# fig.update_layout(yaxis_nticks=54,xaxis_nticks=54)

# Create a list with the days
days = [6, 7, 8, 9, 10, 13, 14, 15, 16, 17]

### ARTICLES PREPROCESSING ###
articles_final = pd.read_csv('articles_final.csv')
articles_final.sort_values(by='article_number',inplace = True)
articles_wordcloud = pd.read_csv('articles_wordcloud.csv')

print(articles_wordcloud['content'])

article_list1 = articles_wordcloud['content'].iloc[1]
article_list1 = article_list1.split()
print(article_list1)

word_list = []
for i in range(len(articles_final)):

    article_list = articles_wordcloud['content'].iloc[i]
    article_list = article_list.split()
    word_list += article_list

word_counts = Counter(word_list)
word_counts = word_counts.most_common(50)

words_bag = []
freqs = []
for i in range(len(word_counts)):
    words_bag.append(word_counts[i][0])
    freqs.append(word_counts[i][1])

print(words_bag, freqs)
freqs = [i * 0.1 for i in freqs]

colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(len(words_bag))]
weights = freqs


color_discrete_map = {'Negative': 'rgb(255,0,0)', 'Positive': 'rgb(0,255,0)', 'Neutral': 'rgb(0,0,255)'}
fig1 = px.scatter(articles_final, x="dates", y="Polarity", color='Sentiment',color_discrete_map=color_discrete_map,hover_data=['article_title'])

data = go.Scatter(x=[random.random() for i in range(len(words_bag))],
                 y=[random.random() for i in range(len(words_bag))],
                 mode='text',
                 text=words_bag,
                 marker={'opacity': 0.3},
                 textfont={'size': weights,'color': colors})
layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                    'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})

fig2 = go.Figure(data=[data], layout=layout)

idx = pd.date_range('01-06-2014', '01-17-2014')

### Dash implementation ###

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # div for image set
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("Engineering", style={'textAlign': 'center'}),
                html.Img(id="Engg", src='data:image/png;base64,{}'.format(test_3), style={'width': '98%'}),
            ]),

            dbc.Col([
                html.H5("Executives", style={'textAlign': 'center'}),
                html.Img(id="Exec", src='data:image/png;base64,{}'.format(test_1), style={'width': '98%'}),
            ]),

            dbc.Col([
                html.H5("Facilities", style={'textAlign': 'center'}),
                html.Img(id="Facil", src='data:image/png;base64,{}'.format(test_4), style={'width': '98%'}),
            ]),

            dbc.Col([
                html.H5("IT", style={'textAlign': 'center'}),
                html.Img(id="itt", src='data:image/png;base64,{}'.format(test_2), style={'width': '98%'}),
            ]),

            dbc.Col([
                html.H5("Security", style={'textAlign': 'center'}),
                html.Img(id="Security", src='data:image/png;base64,{}'.format(test_5), style={'width': '98%'})
            ]),
        ], style={'display': 'flex'})
    ]),

    # div for dropdown set
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='engg',
                options=[{'label': x, 'value': x} for x in Engineering],
                value=[],
                multi=True,
            )
        ], style={'width': '18%', 'padding-left': '2%'},
            className='six columns'),
        html.Div([
            dcc.Dropdown(
                id='exect',
                options=[{'label': x, 'value': x} for x in Executive],
                value=[],
                multi=True,
            )
        ], style={'width': '16%'},
            className='six columns'),
        html.Div([
            dcc.Dropdown(
                id='facil',
                options=[{'label': x, 'value': x} for x in Facilities],
                value=[],
                multi=True,
            )
        ], style={'width': '16%'},
            className='six columns'),
        html.Div([
            dcc.Dropdown(
                id='IT',
                options=[{'label': x, 'value': x} for x in Information_Technology],
                value=[],
                multi=True,
            )
        ],
            style={'width': '16%'},
            className='six columns'),
        html.Div([
            dcc.Dropdown(
                id='security',
                options=[{'label': x, 'value': x} for x in Security],
                value=[],
                multi=True,
            )
        ], style={'width': '16%'},
            className='six columns'),
    ], className='row'),

    # div for adj_martix, subject table, timeline, checkbox, tod filter
    html.Div([
        html.Div([
            html.H4('Emails heatmap'),
            dcc.Graph(id='graph', figure=fig, clickData=None),
            dcc.Dropdown(id='dropdown', options=tod, value=None),
            dcc.Checklist(
                id='checklist',
                options=[
                    {'label': 'Communication between all departments',
                     'value': 'Communication between all departments'},
                    {'label': 'Communication between different departments',
                     'value': 'Communication between different departments'}],
                value=[],
                inline = True
            )
        ], className='six columns'),

        html.Div([
            html.H4('Number of emails'),
            dcc.Graph(id='communication', clickData = None),
        ], className='six columns'),

    ], className='row'),

    html.Div([
    # word cloud
    html.Div([
        html.H4("Email Headers WordCloud"),
        html.Img(id="image_wc", style={'padding': 10})
    ], className='six columns'),

    html.Div([
    html.H4('Emails subjects'),
        dash_table.DataTable(id='subject', page_size=5, style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'width': '80px'
        }, columns=[{'name': i, 'id': i} for i in df_emails_original[['From_Name', 'To_Name','Subject','Date']]], data=getData(df_emails)),
    ], className='six columns')
    ], className='row'),

    # div for articles
    html.Div([
        html.Div([
            html.H4('Sentiment Classification of Articles'),
            dcc.Graph(id='sentiment-graph', style={'width': '100vh', 'height': '75vh'},figure=fig1, clickData=None),
dcc.Checklist(
                id='checklist2',
                options=[
                    {'label': 'Protectors of Kronos',
                     'value': 'Protectors of Kronos'},
                    {'label': 'Sanjorge',
                     'value': 'Sanjorge'},
                    {'label': 'Emergency',
                     'value': 'Emergency'},
                    {'label': 'Voices - 20 January 2014',
                     'value': 'Voices - 20 January 2014'}
                ],
                value=[],
                inline=True
            )
        ],className='six columns'),
        html.Div([
            html.H4('Articles WordCloud'),
            dcc.Graph(id='wordcloud-graph', style={'width': '100vh', 'height': '75vh'}, figure=fig2, clickData=None),
        ],className='six columns')
    ],className='row')
])


@app.callback(
    Output('image_wc', 'src'),
    [Input('image_wc', 'id'),
     Input('graph', 'clickData'),
     Input("dropdown", "value"),
     Input("checklist", "value"),
     Input("engg", "value"),
     Input("exect", "value"),
     Input("facil", "value"),
     Input("IT", "value"),
     Input("security", "value")])
def plot_wordcloud(ids, gra, tod, checklist, engg, exect, facil, IT, security):
    join_list = engg + exect + facil + IT + security

    # ctx = dash.callback_context
    # recent = ctx.triggered[0]['prop_id'].split('.')[0]
    if join_list != []:
        df = final[final['LastName_From'].isin(join_list)]
        df['Subject'] = df['Subject'].str.replace("RE: ", "")
        df['Subject'] = df['Subject'].str.replace("FW: ", "")
        if checklist == ["Communication between different departments"]:
            df = df[df['CurrentEmploymentType_From'] != df['CurrentEmploymentType_To']]
        if tod != None:
            df = df[df['Time_of_day'] == tod]
        if gra != None:
            person_from = gra['points'][0]['y']
            person_to = gra['points'][0]['x']
            if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
                df = df[df['LastName_From'] == person_from]
            else:
                df = df[df['CurrentEmploymentType_From'] == person_from]
            df = df[df['CurrentEmploymentType_To'] == person_to]

        allWords = ' '.join([word for word in df['Subject']])

    elif join_list == [] and gra != None:
        person_from = gra['points'][0]['y']
        person_to = gra['points'][0]['x']
        if (final['CurrentEmploymentType_From'] == person_from).eq(False).all():
            df = final[final['LastName_From'] == person_from]
        else:
            df = final[final['CurrentEmploymentType_From'] == person_from]
        df = final[final['CurrentEmploymentType_From'] == person_to]
        allWords = ' '.join([word for word in df['Subject']])

    else:
        allWords = ' '.join([word for word in final['Subject']])

    wc = WordCloud(width=500, height=300, random_state=21, max_font_size=115).generate(allWords)
    new = wc.to_image()
    img = BytesIO()
    new.save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


@app.callback(
    Output("graph", "figure"),
    [Input("dropdown", "value"),
     Input("checklist", "value"),
     Input("engg","value"),
     Input("exect","value"),
     Input("facil","value"),
     Input("IT","value"),
     Input("security","value")])
def update_adj_marix(tod, checklist,engg, exect, facil, it, security):
    join_list = engg + exect + facil + it + security
    if join_list != []:
        df = final[final['LastName_From'].isin(join_list)]
        if checklist == ["Communication between different departments"]:
            df = df[df['CurrentEmploymentType_From'] != df['CurrentEmploymentType_To']]
        if tod != None:
            df = df[df['Time_of_day'] == tod]
        graphdf_names = df[['LastName_From', 'CurrentEmploymentType_To', 'count']]
        pivot = pd.pivot_table(graphdf_names, values='count', index='LastName_From',
                               columns='CurrentEmploymentType_To',
                               aggfunc='count')

        pivot = pivot.fillna(0)
        fig = px.imshow(pivot, width=600, height=600, color_continuous_scale='gray_r')
        return fig
    else:
        df = final.copy()
        if checklist == ["Communication between different departments"]:
            df = final[final['CurrentEmploymentType_From'] != final['CurrentEmploymentType_To']]
        if tod != None:
            df = df[df['Time_of_day'] == tod]

        graphdf_names = df[['CurrentEmploymentType_From', 'CurrentEmploymentType_To', 'count']]
        pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From',
                               columns='CurrentEmploymentType_To',
                               aggfunc='count')

        pivot = pivot.fillna(0)
        fig = px.imshow(pivot, width=600, height=600, color_continuous_scale='gray_r')
        return fig


@app.callback(
    Output("communication", "figure"),
    [Input("graph", "clickData"),
     Input("engg", "value"),
     Input("exect", "value"),
     Input("facil", "value"),
     Input("IT", "value"),
     Input("security", "value"),
     Input("checklist", "value")
     ])
def update_timeline(people,engg, exect, facil, it, security, checklist):
    ctx = dash.callback_context
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]
    join_list = engg + exect + facil + it + security
    df = final.copy()

    checklist_value = 'all departments'

    if checklist == ["Communication between different departments"]:
        df = final[final['CurrentEmploymentType_From'] != final['CurrentEmploymentType_To']]
        df['dates_count'] = df.groupby('just_date')['just_date'].transform('count')
        checklist_value = 'different departments'

    if people == None and join_list == []:
        if checklist == ["Communication between different departments"]:
            only_dates = df.drop_duplicates(subset='dates_count', keep="first")
            fig = px.bar(only_dates, x="just_date", y="dates_count",barmode= 'relative', title='Communication between different departments')
            return fig
        else:
            only_dates = df.drop_duplicates(subset='dates_count', keep="first")
            fig = px.bar(only_dates, x="just_date", y="dates_count",barmode= 'relative', title='Communication between all departments')
            return fig

    if clicked_element == 'graph':
        person_from = people['points'][0]['y']
        person_to = people['points'][0]['x']
        if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
            dff = df[df['LastName_From'] == person_from]
        else:
            dff = df[df['CurrentEmploymentType_From'] == person_from]

        final_df = dff[dff['CurrentEmploymentType_To'] == person_to]

        final_df['dates_count'] = final_df.groupby('just_date')['just_date'].transform('count')
        only_dates = final_df.drop_duplicates(subset='just_date', keep="first")
        only_dates = only_dates[['just_date', 'dates_count']]
        only_dates.set_index('just_date', inplace=True)
        only_dates.index = pd.DatetimeIndex(only_dates.index)

        only_dates = only_dates.reindex(idx, fill_value=0)

        only_dates.reset_index(inplace=True)

        fig = px.bar(only_dates, x="index", y="dates_count", barmode='relative',
                     title=f'Communication: From: {person_from} To {person_to}')

        return fig
    else:
        if join_list == []:
            only_dates = df.drop_duplicates(subset='dates_count', keep="first")
            fig = px.bar(only_dates, x="just_date", y="dates_count", barmode='relative', title=f'Communication {checklist_value}')
            return fig
        else:
            df = df[df['LastName_From'].isin(join_list)]
            df['dates_count'] = df.groupby('just_date')['just_date'].transform('count')
            only_dates = df.drop_duplicates(subset='just_date', keep="first")

            only_dates= only_dates[['just_date','dates_count']]

            only_dates.set_index('just_date',inplace=True)

            only_dates.index = pd.DatetimeIndex(only_dates.index)

            only_dates = only_dates.reindex(idx, fill_value=0)

            only_dates.reset_index(inplace=True)


            fig = px.bar(only_dates, x="index", y="dates_count", barmode= 'relative',
                          title=f'Communication: From: {join_list} to {checklist_value}')

            return fig

@app.callback(
    Output("subject", "data"),
    [Input("graph", "clickData"),
     Input("dropdown", "value"),
     Input("checklist", "value"),
     Input("communication", "clickData"),
     Input("engg", "value"),
     Input("exect", "value"),
     Input("facil", "value"),
     Input("IT", "value"),
     Input("security", "value")
     ])
def update_subject(people, tod, checklist, communication, engg, exect, facil, it, security):
    ctx = dash.callback_context
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]
    join_list = engg + exect + facil + it + security
    # if checklist == ["Communication between different departments"]:
    #     df = df[df['CurrentEmploymentType_From'] != df['CurrentEmploymentType_To']]
    if clicked_element in ['engg', 'exect', 'facil', 'IT', 'security']:
        print("befooore")
        print(clicked_element)
        print(join_list)
        df = merge_original_from[merge_original_from['LastName_From'].isin(join_list)]
        if tod != None:
            final_df = df[df['Time_of_day'] == tod]
            return getData(final_df)
        else:
            return getData(df)
    elif clicked_element == 'graph':
        if tod != None:
            df = merge_original_from[merge_original_from['Time_of_day'] == tod]
            person_from = people['points'][0]['y']
            person_to = people['points'][0]['x']
            if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
                dff = df[df['LastName_From'] == person_from]
            else:
                dff = df[df['CurrentEmploymentType_From'] == person_from]
            final_df_values = dff['CurrentEmploymentType_To_List'].apply(lambda employee: person_to in employee)
            final_df = dff[final_df_values]

            return getData(final_df)
        else:
            person_from = people['points'][0]['y']
            person_to = people['points'][0]['x']
            if (merge_original_from['CurrentEmploymentType_From'] == person_from).eq(False).all():
                dff = merge_original_from[merge_original_from['LastName_From'] == person_from]
            else:
                dff = merge_original_from[merge_original_from['CurrentEmploymentType_From'] == person_from]
            # dff = df[df['CurrentEmploymentType_From'].str.contains(person_from)]
            final_df_values = dff['CurrentEmploymentType_To_List'].apply(lambda employee: person_to in employee)
            final_df = dff[final_df_values]

            return getData(final_df)
    else:
        if people != None:
            if tod != None:
                df = merge_original_from[merge_original_from['Time_of_day'] == tod]
                person_from = people['points'][0]['y']
                person_to = people['points'][0]['x']
                if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
                    dff = df[df['LastName_From'] == person_from]
                else:
                    dff = df[df['CurrentEmploymentType_From'] == person_from]
                final_df_values = dff['CurrentEmploymentType_To_List'].apply(lambda employee: person_to in employee)
                final_df = dff[final_df_values]

                return getData(final_df)
            else:
                person_from = people['points'][0]['y']
                person_to = people['points'][0]['x']
                if (merge_original_from['CurrentEmploymentType_From'] == person_from).eq(False).all():
                    dff = merge_original_from[merge_original_from['LastName_From'] == person_from]
                else:
                    dff = merge_original_from[merge_original_from['CurrentEmploymentType_From'] == person_from]
                # dff = df[df['CurrentEmploymentType_From'].str.contains(person_from)]
                final_df_values = dff['CurrentEmploymentType_To_List'].apply(lambda employee: person_to in employee)
                final_df = dff[final_df_values]

                return getData(final_df)
        else:
            if tod != None:
                return getData(merge_original_from[merge_original_from['Time_of_day'] == tod])
            else:
                return getData(merge_original_from)
@app.callback(
    [Output("sentiment-graph", "figure"),
     Output("wordcloud-graph", "figure")],
    [Input("checklist2", "value"),
     Input("sentiment-graph", "clickData")]
)
def update_articles(checklist_values,clickData):
    ctx = dash.callback_context
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]

    if clicked_element == "checklist2":

        if checklist_values:
            articles_filtered = articles_final[articles_final['cluster'].isin(checklist_values)]
        else:
            articles_filtered = articles_final

        color_discrete_map = {'Negative': 'rgb(255,0,0)', 'Positive': 'rgb(0,255,0)', 'Neutral': 'rgb(0,0,255)'}
        fig1 = px.scatter(articles_filtered, x="dates", y="Polarity", color='Sentiment',
                          color_discrete_map=color_discrete_map,
                          hover_data=['article_number', 'article_title'])

        article_num_list = list(articles_filtered['article_number'])
        articles_wc = articles_wordcloud[articles_wordcloud['article_number'].isin(article_num_list)]
        word_list = []
        for i in range(len(articles_wc)):
            article_list = articles_wc['content'].iloc[i]
            article_list = article_list.split()
            word_list += article_list

        word_counts = Counter(word_list)
        word_counts = word_counts.most_common(60)

        words_bag = []
        freqs = []
        for i in range(len(word_counts)):
            words_bag.append(word_counts[i][0])
            freqs.append(word_counts[i][1])

        lower, upper = 10, 60
        freqs = [((x - min(freqs)) / (max(freqs) - min(freqs))) * (upper - lower) + lower for x in freqs]

        colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(len(words_bag))]
        weights = freqs

        data = go.Scatter(x=[random.uniform(0, 1) for i in range(len(words_bag))],
                          y=[random.uniform(0, 1) for i in range(len(words_bag))],
                          mode='text',
                          text=words_bag,
                          marker={'opacity': 0.3},
                          textfont={'size': weights, 'color': colors})
        layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})

        fig2 = go.Figure(data=[data], layout=layout)

        return fig1, fig2
    else:

        if checklist_values:

            articles_filtered = articles_final[articles_final['cluster'].isin(checklist_values)]

            if clickData is not None:

                click_article_num = clickData['points'][0]['customdata'][0]
                click_article_title = clickData['points'][0]['customdata'][1]
                articles_wc = articles_wordcloud[articles_wordcloud['article_number'] == click_article_num]

            else:

                article_num_list = list(articles_filtered['article_number'])
                articles_wc = articles_wordcloud[articles_wordcloud['article_number'].isin(article_num_list)]

        else:

            articles_filtered = articles_final

            if clickData is not None:

                click_article_num = clickData['points'][0]['customdata'][0]
                click_article_title = clickData['points'][0]['customdata'][1]
                articles_wc = articles_wordcloud[articles_wordcloud['article_number'] == click_article_num]

            else:

                article_num_list = list(articles_filtered['article_number'])
                articles_wc = articles_wordcloud[articles_wordcloud['article_number'].isin(article_num_list)]


        color_discrete_map = {'Negative': 'rgb(255,0,0)', 'Positive': 'rgb(0,255,0)', 'Neutral': 'rgb(0,0,255)'}
        fig1 = px.scatter(articles_filtered, x="dates", y="Polarity", color='Sentiment', color_discrete_map=color_discrete_map,
                              hover_data=['article_number','article_title'])


        word_list = []
        for i in range(len(articles_wc)):
            article_list = articles_wc['content'].iloc[i]
            article_list = article_list.split()
            word_list += article_list

        word_counts = Counter(word_list)
        word_counts = word_counts.most_common(60)

        words_bag = []
        freqs = []
        for i in range(len(word_counts)):
            words_bag.append(word_counts[i][0])
            freqs.append(word_counts[i][1])

        lower, upper = 10, 60
        freqs = [((x - min(freqs)) / (max(freqs) - min(freqs))) * (upper - lower) + lower for x in freqs]

        colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(len(words_bag))]
        weights = freqs

        data = go.Scatter(x=[random.uniform(0, 1) for i in range(len(words_bag))],
                          y=[random.uniform(0, 1) for i in range(len(words_bag))],
                          mode='text',
                          text=words_bag,
                          marker={'opacity': 0.3},
                          textfont={'size': weights, 'color': colors})
        layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})

        fig2 = go.Figure(data=[data], layout=layout)

        return fig1,fig2
if __name__ == '__main__':
    app.run_server()
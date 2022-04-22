#!/usr/bin/env python
# coding: utf-8

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
from ast import literal_eval
import re
from dash_iconify import DashIconify

# Function needed in order to update the values in the table
def getData(df):
    return df.to_dict('rows')

# Read employee records from csv
df_employees = pd.read_excel('csvs/EmployeeRecords.xlsx')

# Import the appropriate images to create the organograms at the top of the dashboard
executives_png = 'pngs/exec.png'
IT_png = 'pngs/IT.png'
Engg_png = 'pngs/Engg.png'
Fac_png = 'pngs/fac.png'
Sec_png = 'pngs/security.png'
test_1 = base64.b64encode(open(executives_png, 'rb').read()).decode('ascii')
test_2 = base64.b64encode(open(IT_png, 'rb').read()).decode('ascii')
test_3 = base64.b64encode(open(Engg_png, 'rb').read()).decode('ascii')
test_4 = base64.b64encode(open(Fac_png, 'rb').read()).decode('ascii')
test_5 = base64.b64encode(open(Sec_png, 'rb').read()).decode('ascii')

# Create employees list
dept = df_employees['CurrentEmploymentType'].unique().tolist()
for i in dept:
    globals()[f"{i}"] = df_employees.loc[df_employees['CurrentEmploymentType'] == i, 'LastName'].tolist()

# Remove stopwords for subject
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('of')
stopwords.append('hey')

# Import the csv files with emails and employeeRecords preprocessed
final = pd.read_csv('csvs/final.csv',index_col=[0])
merge_original_from = pd.read_csv('csvs/merge_original_from.csv',index_col=[0])


# Import the preprocessed datasets for articles
articles_final = pd.read_csv('csvs/articles_final.csv')
articles_final.sort_values(by='article_number',inplace = True)
articles_wordcloud = pd.read_csv('csvs/articles_wordcloud.csv')

article_list1 = articles_wordcloud['content'].iloc[1]
article_list1 = article_list1.split()

# Get each word to create the wordcloud
word_list = []
for i in range(len(articles_final)):

    article_list = articles_wordcloud['content'].iloc[i]
    article_list = article_list.split()
    word_list += article_list

# Show only 50 words
word_counts = Counter(word_list)
word_counts = word_counts.most_common(50)

# Find the frequency of the words
words_bag = []
freqs = []
for i in range(len(word_counts)):
    words_bag.append(word_counts[i][0])
    freqs.append(word_counts[i][1])

# Multiply each freq with 0.1 to create the weights
freqs = [i * 0.1 for i in freqs]

colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(len(words_bag))]
weights = freqs

# Create the scatterplot figure
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

# Read txt file:
text_markdown = "\t"
with open('articles/default.txt') as this_file:
    for a in this_file.read():
        if "\n" in a:
            text_markdown += "\n \t"
        else:
            text_markdown += a

""" Dash Implementation """

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# Css for tab style
tabs_styles = {
    'height': '44px',
    'margin-bottom': '20px',
    'backgroundColor': '#d3dedc'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': '#FFFFFF'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'backgroundColor': '#218173',
    'color': 'white',
    'padding': '6px'
}

# Css for tab style
background = {
    'position': 'absolute',
    'top': '0px',
    'left': '0px',
    'width': '100%',
    'height': '30%',
    'background-color': '#218173'
}

# Form the layout of the dashboard

# Create the organogram and the filter space below this
app.title = 'Disappearance at GASTech'
app.layout = html.Div([
html.Div([
    html.Div(html.H1("Disappearance at GASTech"), style={'textAlign': 'center', 'color': '#FFFFFF'}),
    # Div for image set and filters
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("Engineering", style={'textAlign': 'center'}),
                html.Div([
                html.Img(id="Engg", src='data:image/png;base64,{}'.format(test_3), style={'width': '98%'})], style={'height': '60%'}),
                html.Div([
                dcc.Dropdown(
                    id='engg',
                    options=[{'label': x, 'value': x} for x in Engineering],
                    value=[],
                    multi=True,
                )], style={'width': '98%', 'margin':'auto',
                                'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                                'background-color': '#FFFFFF'
                                })
            ]
            ),

            dbc.Col([
                html.H5("Executives", style={'textAlign': 'center'}),
                html.Div([
                html.Img(id="Exec", src='data:image/png;base64,{}'.format(test_1), style={'width': '98%'})], style={'height': '60%'}),
                html.Div([
                dcc.Dropdown(
                    id='exect',
                    options=[{'label': x, 'value': x} for x in Executive],
                    value=[],
                    multi=True,
                )],style={'width': '98%', 'margin':'auto',
                                'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                                'background-color': '#FFFFFF'
                                })
            ]
            ),

            dbc.Col([
                html.H5("Facilities", style={'textAlign': 'center'}),
                html.Div([
                html.Img(id="Facil", src='data:image/png;base64,{}'.format(test_4), style={'width': '98%'})], style={'height': '60%'}),
                html.Div([
                dcc.Dropdown(
                    id='facil',
                    options=[{'label': x, 'value': x} for x in Facilities],
                    value=[],
                    multi=True,
                )], style={'width': '98%', 'margin':'auto',
                                'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                                'background-color': '#FFFFFF'
                                })
            ]
            ),

            dbc.Col([
                html.H5("IT", style={'textAlign': 'center'}),
                html.Div([
                html.Img(id="itt", src='data:image/png;base64,{}'.format(test_2), style={'width': '98%'})], style={'height': '60%'}),
                html.Div([
                dcc.Dropdown(
                    id='IT',
                    options=[{'label': x, 'value': x} for x in IT],
                    value=[],
                    multi=True,
                )], style={'width': '98%', 'margin':'auto',
                            'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                            'background-color': '#FFFFFF'})
            ]
            ),

            dbc.Col([
                html.H5("Security", style={'textAlign': 'center'}),
                html.Div([
                html.Img(id="Security", src='data:image/png;base64,{}'.format(test_5), style={'width': '98%'})], style={'height': '60%'}),
                html.Div([
                dcc.Dropdown(
                    id='security',
                    options=[{'label': x, 'value': x} for x in Security],
                    value=[],
                    multi=True,
                )
                ], style={'width': '98%', 'margin':'auto',
                                'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                                'background-color': '#FFFFFF'
                                })
            ]
            ),
        ], style={'display': 'flex'})
    ], style={'width': '98%', 'margin': 'auto',
              'border-radius': '10px',
              'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
              'background-color': '#FFFFFF',
              'margin-bottom': '5px'
              }
    )],style={'margin-bottom': '20px'}),
    # Div for tabs
    html.Div([
        dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
            dcc.Tab(label='Communication', value='tab-1', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Articles', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        ], style=tabs_styles),
        html.Div(id='tabs-content-inline')
    ]),
],style = background)


@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    """ This function create the content of each tab. It is split in Communication (tab-1) and Articles (tab-2)
    """
    if tab == 'tab-1':
        return html.Div([
            # Div for adj_martix, subject table, timeline, checkbox, tod filter
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            DashIconify(
                                icon="carbon:chart-network",
                                width=40,
                                style={'margin-right': '5px'}
                            ),
                            html.H5('Emails Adjacency Matrix')
                        ], style={'display': 'flex'}),
                        dcc.Graph(id='graph', clickData=None),

                    ], style={'width': '70%', 'height': '90%',
                              'display': 'inline-block',
                              'border-radius': '10px',
                              'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                              'background-color': '#FFFFFF',
                              'padding': '5px',
                              'margin-top': '10px',
                              'margin-bottom': '5px',
                              'margin-left': '5px'}),  # className='three columns'),

                    dbc.Col([
                        html.Div([
                            DashIconify(
                                icon="clarity:email-line",
                                width=40,
                                style={'margin-right': '5px'}
                            ),
                            html.H5('Email Frequency')
                        ], style={'display': 'flex'}),
                        dcc.Graph(id='communication'),
                    ], style={'width': '70%', 'height': '90%',
                              'display': 'inline-block',
                              'border-radius': '10px',
                              'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                              'background-color': '#FFFFFF',
                              'padding': '5px',
                              'margin-top': '10px',
                              'margin-bottom': '5px',
                              'margin-left': '10px'}),  # className='three columns'),
                    dbc.Col([
                        dbc.Row([
                            html.Br(),
                            html.Br(),
                            html.H5('Inter-Intra Departments'),
                            dcc.RadioItems(
                                id='checklist',
                                options=[
                                    {'label': 'All departments',
                                     'value': 'Communication between all departments'},
                                    {'label': 'Inter departments',
                                     'value': 'Communication between different departments'}],
                                value="Communication between all departments",
                                labelStyle={'display': 'inline-block'}
                            )
                        ], style={'width': '85%', 'height': '44%',
                                  #                    'display': 'inline-block',
                                  #                    'align-items': 'center',
                                  #                   'justify-content': 'center',
                                  'text-align': 'center',
                                  #                   'justify-content': 'left',
                                  'border-radius': '10px',
                                  'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                                  'background-color': '#FFFFFF',
                                  'padding': '10px',
                                  'margin-top': '10px',
                                  'margin-bottom': '10px'}),
                        #             style={‘width’: ‘100%’, ‘display’: ‘flex’, ‘align-items’:‘center’, ‘justify-content’:‘center’}), #, className='three columns' ),
                        dbc.Row([
                            html.Br(),
                            html.Br(),
                            html.H5('Mail Hour Tracking'),
                            dcc.RadioItems(id='dropdown', options=[
                                {'label': 'Working hours (9-5)', 'value': 'Working hours (9-5)'},
                                {'label': 'After hours (the rest)', 'value': 'After hours (the rest)'},
                                {'label': 'All day', 'value': 'both'}],
                                           value='both',
                                           labelStyle={'display': 'inline-block'}
                                           ),

                        ], style={'width': '85%', 'height': '44%',
                                  'text-align': 'center',
                                  #                   'display': 'inline-block',
                                  'border-radius': '10px',
                                  'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                                  'background-color': '#FFFFFF',
                                  'padding': '10px',
                                  'margin-top': '5px'}),  # , className='three columns')
                    ], style={'margin-left': '10px'}),
                ], style={'display': 'flex'})
            ]),

            html.Div([
                # Div for wordcloud
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            DashIconify(
                                icon="fluent:cloud-words-32-regular",
                                width=40,
                                style={'margin-right': '5px'}
                            ),
                            html.H5('Email Headers Wordcloud')
                        ], style={'display': 'flex'}),
                        html.Img(id="image_wc", style={'padding': 10}),
                        dcc.RadioItems(
                            id='radio-items',
                            options=[
                                {'label': 'Most common', 'value': 'common'},
                                {'label': 'Least common', 'value': 'not_common'}
                            ],
                            value="",
                            labelStyle={'display': 'inline-block'}
                        )], style={'width': '41%',
                                   #                                'display': 'inline-block',
                                   'border-radius': '10px',
                                   'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                                   'background-color': '#FFFFFF',
                                   'margin-bottom': '5px',
                                   'margin-right': '10px'}),

                    dbc.Col([
                        html.Div([
                            DashIconify(
                                icon="icon-park-outline:topic",
                                width=40,
                                style={'margin-right': '5px'}
                            ),
                            html.H5('Email Subjects')
                        ], style={'display': 'flex'}),
                        dash_table.DataTable(id='subject', page_size=5, style_data={
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'width': '80px',
                            'color': 'black',
                            'backgroundColor': 'rgb(234,239,240)'
                        },  style_header={
                            'backgroundColor': 'rgb(57,134,129)',
                            'color': 'white',
                            'fontWeight': 'bold'
                        }, columns=[{'name': i, 'id': i} for i in
                                    merge_original_from[['From_Name', 'To_Name', 'Subject', 'Date']]],
                                             data=getData(merge_original_from)),
                    ], style={'width': '57%', 'display': 'inline-block',
                              'border-radius': '10px',
                              'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                              'background-color': '#FFFFFF',
                              'padding': '5px',
                              'margin-bottom': '5px'})
                ], style={'display': 'flex'})
            ])
        ])
    elif tab == 'tab-2':
        # Div for articles
        return html.Div([
            html.Br(),
            html.Div([
                html.Div([
                    html.Div([
                        DashIconify(
                            icon="ic:baseline-sentiment-satisfied",
                            width=40,
                            style={'margin-right': '5px'}
                        ),
                        html.H5('Sentiment Classification of Articles')
                    ], style={'display': 'flex'}),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(id='sentiment-graph', figure=fig1,
                              clickData=None, style={'width': '90%', 'height': '90%',
                                                     'border-radius': '10px',
                                                     'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                                                     'background-color': '#FFFFFF',
                                                     'padding': '5px',
                                                     'margin-left': '5px',
                                                     'margin-top': '5px',
                                                     'margin-bottom': '5px'}),
                    dcc.Checklist(
                        id='checklist2',
                        options=[
                            {'label': 'Bag-of-Words-1: [Tethys, Government, Sanjorge, Health, Gastech]',
                             'value': 'C0'},
                            {'label': 'Bag-of-Words-2: [Emergency, Voices, Construction, Fire, Scene]',
                             'value': 'C1'},
                            {'label': 'Bag-of-Words-3: [Protesters, Protectors of Kronos(POK), Police, Karel, Force]',
                             'value': 'C2'}
                        ],
                        value=[],
                        labelStyle={'display': 'inline-block'},
                        style={'width': '90%', 'height': '90%',
                               'border-radius': '10px',
                               'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                               'background-color': '#FFFFFF',
                               'padding': '5px',
                               'margin-left': '5px',
                               'margin-top': '5px',
                               'margin-bottom': '5px'}
                    )
                ], className='six columns'),
                html.Div([
                    html.Div([
                        DashIconify(
                            icon="fluent:cloud-words-32-regular",
                            width=40,
                            style={'margin-right': '5px'}
                        ),
                        html.H5('Articles WordCloud')
                    ], style={'display': 'flex'}),
                    html.Br(),
                    html.Br(),
                    html.Img(id="articles_wc", style={'width': '90%', 'height': '90%',
                                                      'border-radius': '10px',
                                                      'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                                                      'background-color': '#FFFFFF',
                                                      'padding': '5px',
                                                      'margin-left': '5px',
                                                      'margin-top': '5px',
                                                      'margin-bottom': '5px'})
                ], className='six columns'),
            ], className='row'),

            html.Br(),
            html.Br(),
            html.Div([
                html.Div([
                    DashIconify(icon="ic:twotone-article", width=40),
                    html.H5("Selected Article Content", className="card-title"),
                ], style={'display': 'flex'}),
                dcc.Markdown(text_markdown, id='article-container'),
            ], style={'width': '98%', 'height': '90%',
                      'border-radius': '5px',
                      'box-shadow': 'rgba(0, 0, 0, 0.24) 0px 3px 8px',
                      'background-color': '#FFFFFF',
                      'padding': '5px',
                      'margin-right': '10px',
                      'margin-left': '10px',
                      'margin-top': '5px',
                      'margin-bottom': '5px'})
        ])

@app.callback(
    Output('image_wc', 'src'),
    [Input('graph', 'clickData'),
     Input("dropdown", "value"),
     Input("checklist", "value"),
     Input("engg", "value"),
     Input("exect", "value"),
     Input("facil", "value"),
     Input("IT", "value"),
     Input("security", "value"),
     Input('radio-items', 'value'),
     Input('communication', 'clickData')])
def plot_wordcloud(gra, tod, checklist, engg, exect, facil, IT, security, radio, comm):
    """ This function used for creating and updating the wordcloud by applying the above input
    as filters
    """

    join_list = engg + exect + facil + IT + security

    if join_list != []:
        df = merge_original_from[merge_original_from['LastName_From'].isin(join_list)]

        if checklist == "Communication between different departments":
            df = df[df['CurrentEmploymentType_From'] != df['CurrentEmploymentType_To_Unlisted']]
        if tod != 'both':
            df = df[df['Time_of_day'] == tod]
        if comm != None:
            # startdate = pd.to_datetime(comm['points'][0]['x']).date()
            df = df.loc[df['just_date'] == comm['points'][0]['x']]
        #             df = df[df['just_date'] == comm['points'][0]['x']]
        if gra != None:
            person_from = gra['points'][0]['y']
            person_to = gra['points'][0]['x']
            if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
                df = df[df['LastName_From'] == person_from]
            else:
                df = df[df['CurrentEmploymentType_From'] == person_from]
            final_df_values = df['CurrentEmploymentType_To_Unique'].apply(lambda employee: person_to in employee)

            df = df[final_df_values]
        allWords = ' '.join([word for word in df['Subject_new']])

    elif join_list == [] and gra != None:

        person_from = gra['points'][0]['y']
        person_to = gra['points'][0]['x']
        if (merge_original_from['CurrentEmploymentType_From'] == person_from).eq(False).all():
            df = merge_original_from[merge_original_from['LastName_From'] == person_from]
        else:

            df = merge_original_from[merge_original_from['CurrentEmploymentType_From'] == person_from]

        final_df_values = df['CurrentEmploymentType_To_Unique'].apply(lambda employee: person_to in employee)
        df = df[final_df_values]
        if comm != None:
            # startdate = pd.to_datetime(comm['points'][0]['x']).date()
            df = df.loc[df['just_date'] == comm['points'][0]['x']]
        if checklist == "Communication between different departments":
            df = df[df['CurrentEmploymentType_From'] != df['CurrentEmploymentType_To_Unlisted']]
        if tod != 'both':
            df = df[df['Time_of_day'] == tod]

        allWords = ' '.join([word for word in df['Subject_new']])

    else:
        df = merge_original_from.copy()

        if comm != None:
            df = df.loc[df['just_date']==comm['points'][0]['x']]
        if checklist == "Communication between different departments":
            df = df[df['CurrentEmploymentType_From'] != df['CurrentEmploymentType_To_Unlisted']]
        if tod != 'both':
            df = df[df['Time_of_day'] == tod]
        allWords = ' '.join([word for word in df['Subject_new']])

    if radio == 'not_common':
        wc = WordCloud(width=500, height=300, max_words=60, random_state=21, max_font_size=115, background_color='white',
                       colormap='winter').generate(allWords)
        keys = list(wc.words_.keys())
        values = list(wc.words_.values())
        values.reverse()
        wc.words_ = dict(zip(keys, values))
        wc.generate_from_frequencies(wc.words_)

    else:
        wc = WordCloud(width=500, height=300, random_state=21, max_words=60, max_font_size=115, background_color='white',
                       colormap='winter').generate(allWords)

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
    """ This function used for creating and updating the adjacency matrix by applying the above input
    as filters
    """

    join_list = engg + exect + facil + it + security
    if join_list != []:
        df = final[final['LastName_From'].isin(join_list)]
        if checklist == "Communication between different departments":
            df = df[df['CurrentEmploymentType_From'] != df['CurrentEmploymentType_To']]
        if tod != 'both':
            df = df[df['Time_of_day'] == tod]
        graphdf_names = df[['LastName_From', 'CurrentEmploymentType_To', 'count']]
        pivot = pd.pivot_table(graphdf_names, values='count', index='LastName_From',
                               columns='CurrentEmploymentType_To',
                               aggfunc='count')

        pivot = pivot.fillna(0)
        fig = px.imshow(pivot, color_continuous_scale='tempo')
        fig.update_layout(
            title='Number of emails',
            xaxis_title="To",
            yaxis_title="From"
        )
        return fig
    else:
        df = final.copy()
        if checklist == "Communication between different departments":
            df = df[df['CurrentEmploymentType_From'] != df['CurrentEmploymentType_To']]
        if tod != 'both':
            df = df[df['Time_of_day'] == tod]

        graphdf_names = df[['CurrentEmploymentType_From', 'CurrentEmploymentType_To', 'count']]
        pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From',
                               columns='CurrentEmploymentType_To',
                               aggfunc='count')

        pivot = pivot.fillna(0)
        fig = px.imshow(pivot, color_continuous_scale='tempo')
        fig.update_layout(
            title='Number of emails',
            xaxis_title="To",
            yaxis_title="From"
        )
        return fig


@app.callback(
    Output("communication", "figure"),
    [Input("graph", "clickData"),
     Input("engg", "value"),
     Input("exect", "value"),
     Input("facil", "value"),
     Input("IT", "value"),
     Input("security", "value"),
     Input('dropdown', "value"),
     Input("checklist", "value"),
     Input('communication', "clickData")
     ])
def update_timeline(people,engg, exect, facil, it, security, tod, checklist, comm):
    """ This function used for creating and updating the communication timeline by applying the above input
    as filters
    """

    ctx = dash.callback_context
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]
    join_list = engg + exect + facil + it + security
    df = final.copy()

    checklist_value = 'all departments'

    if checklist == "Communication between different departments":
        df = df[df['CurrentEmploymentType_From'] != df['CurrentEmploymentType_To']]
        checklist_value = 'different departments'

    if tod != 'both':
        df = df[df['Time_of_day'] == tod]

    if people == None and join_list == []:
        df['dates_count'] = df.groupby('just_date')['just_date'].transform('count')
        only_dates = df.drop_duplicates(subset='dates_count', keep="first")

        colors = []
        if comm != None:
            for value in only_dates['just_date']:
                if comm['points'][0]['x'] == value:
                    colors.append('#d3dedc')
                else:
                    colors.append('#218173')
        else:
            for i in only_dates['just_date']:
                colors.append('#218173')

        fig = px.bar(only_dates, x="just_date", y="dates_count",barmode= 'relative',color = colors, color_discrete_map="identity")
        fig.update_layout(
            title=f'Communication between {checklist_value}',
            xaxis_title="Dates",
            yaxis_title="Number of emails" , #  paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        return fig

    if clicked_element == 'communication':
        if tod != 'both':
            df = df[df['Time_of_day'] == tod]
        if join_list != []:
            df = df[df['LastName_From'].isin(join_list)]
            person_from = join_list
            person_to = checklist_value
            df['dates_count'] = df.groupby('just_date')['just_date'].transform('count')
            only_dates = df.drop_duplicates(subset='just_date', keep="first")

            only_dates = only_dates[['just_date', 'dates_count']]

            only_dates.set_index('just_date', inplace=True)

            only_dates.index = pd.DatetimeIndex(only_dates.index)

            only_dates = only_dates.reindex(idx, fill_value=0)

            only_dates.reset_index(inplace=True)

            colors = []
            for value in only_dates['index']:
                if comm['points'][0]['x'] == str(value.date()):
                    colors.append('#d3dedc')
                else:
                    colors.append('#218173')

            fig = px.bar(only_dates, x="index", y="dates_count", barmode='relative', color=colors,
                         color_discrete_map="identity")

            fig.update_layout(
                title=f'Communication: From: {person_from} To {person_to}',
                xaxis_title="Dates",
                yaxis_title="Number of emails",  # paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
            return fig
        if people!= None:
            person_from = people['points'][0]['y']
            person_to = people['points'][0]['x']
            if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
                dff = df[df['LastName_From'] == person_from]
            else:
                dff = df[df['CurrentEmploymentType_From'] == person_from]

            df = dff[dff['CurrentEmploymentType_To'] == person_to]

            df['dates_count'] = df.groupby('just_date')['just_date'].transform('count')
            only_dates = df.drop_duplicates(subset='just_date', keep="first")

            only_dates = only_dates[['just_date', 'dates_count']]

            only_dates.set_index('just_date', inplace=True)

            only_dates.index = pd.DatetimeIndex(only_dates.index)

            only_dates = only_dates.reindex(idx, fill_value=0)

            only_dates.reset_index(inplace=True)

            colors = []
            for value in only_dates['index']:
                if comm['points'][0]['x'] == str(value.date()):
                    colors.append('#d3dedc')
                else:
                    colors.append('#218173')

            fig = px.bar(only_dates, x="index", y="dates_count", barmode='relative', color=colors,
                         color_discrete_map="identity")

            fig.update_layout(
                title=f'Communication: From: {person_from} To {person_to}',
                xaxis_title="Dates",
                yaxis_title="Number of emails" , #  paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
            return fig

    if clicked_element == 'graph':
        if tod != 'both':
            df = df[df['Time_of_day'] == tod]

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

        colors = []

        for i in only_dates['index']:
            colors.append('#218173')

        fig = px.bar(only_dates, x="index", y="dates_count", barmode='relative',color = colors, color_discrete_map="identity")

        fig.update_layout(
            title=f'Communication: From: {person_from} To {person_to}',
            xaxis_title="Dates",
            yaxis_title="Number of emails" , #  paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        return fig

    else:
        if tod != 'both':
            df = df[df['Time_of_day'] == tod]
        if join_list == []:
            if people != None and comm != None:
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

                colors = []
                for value in only_dates['index']:
                    if comm['points'][0]['x'] == str(value.date()):
                        colors.append('#d3dedc')
                    else:
                        colors.append('#218173')

                fig = px.bar(only_dates, x="index", y="dates_count", barmode='relative', color=colors,
                             color_discrete_map="identity")

                fig.update_layout(
                    title=f'Communication: From: {person_from} To {person_to}',
                    xaxis_title="Dates",
                    yaxis_title="Number of emails" , #  paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
                return fig
            elif people != None:
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

                colors = []

                for i in only_dates['index']:
                    colors.append('#218173')

                fig = px.bar(only_dates, x="index", y="dates_count", barmode='relative', color=colors,
                             color_discrete_map="identity")

                fig.update_layout(
                    title=f'Communication: From: {person_from} To {person_to}',
                    xaxis_title="Dates",
                    yaxis_title="Number of emails",  # paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
                return fig
            else:

                only_dates = df.drop_duplicates(subset='dates_count', keep="first")

                colors = []
                if comm != None:
                    for value in only_dates['just_date']:
                        if comm['points'][0]['x'] == value:
                            colors.append('#d3dedc')
                        else:
                            colors.append('#218173')
                else:
                    for i in only_dates['just_date']:
                        colors.append('#218173')

                fig = px.bar(only_dates, x="just_date", y="dates_count", barmode='relative',color = colors, color_discrete_map="identity")
                fig.update_layout(
                    title = f'Communication {checklist_value}',
                    xaxis_title="Dates",
                    yaxis_title="Number of emails" , #  paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
                return fig
        else:
            if clicked_element in ['engg', 'exect', 'facil', 'IT', 'security']:
                if join_list == []:
                    join_list = ''
                else:
                    df = df[df['LastName_From'].isin(join_list)]
                df['dates_count'] = df.groupby('just_date')['just_date'].transform('count')
                only_dates = df.drop_duplicates(subset='just_date', keep="first")

                only_dates= only_dates[['just_date','dates_count']]

                only_dates.set_index('just_date',inplace=True)

                only_dates.index = pd.DatetimeIndex(only_dates.index)

                only_dates = only_dates.reindex(idx, fill_value=0)

                only_dates.reset_index(inplace=True)

                colors = []
                for i in only_dates['index']:
                    colors.append('#218173')


                fig = px.bar(only_dates, x="index", y="dates_count", barmode= 'relative',color = colors, color_discrete_map="identity")

                fig.update_layout(
                    title=f'Communication: From: {join_list} to {checklist_value}',
                    xaxis_title="Dates",
                    yaxis_title="Number of emails" , #  paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
                return fig
            elif join_list != [] and comm!=None:
                df = df[df['LastName_From'].isin(join_list)]
                df['dates_count'] = df.groupby('just_date')['just_date'].transform('count')
                only_dates = df.drop_duplicates(subset='just_date', keep="first")

                only_dates = only_dates[['just_date', 'dates_count']]

                only_dates.set_index('just_date', inplace=True)

                only_dates.index = pd.DatetimeIndex(only_dates.index)

                only_dates = only_dates.reindex(idx, fill_value=0)

                only_dates.reset_index(inplace=True)

                colors = []
                for value in only_dates['index']:
                    if comm['points'][0]['x'] == str(value.date()):
                        colors.append('#d3dedc')
                    else:
                        colors.append('#218173')

                fig = px.bar(only_dates, x="index", y="dates_count", barmode='relative', color=colors,
                             color_discrete_map="identity")

                fig.update_layout(
                    title=f'Communication: From: {join_list} to {checklist_value}',
                    xaxis_title="Dates",
                    yaxis_title="Number of emails" , #  paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
                return fig

            else:
                only_dates = df.drop_duplicates(subset='dates_count', keep="first")

                colors = []
                if comm != None:
                    for value in only_dates['just_date']:
                        if comm['points'][0]['x'] == value:
                            colors.append('#d3dedc')
                        else:
                            colors.append('#218173')
                else:
                    for i in only_dates['just_date']:
                        colors.append('#218173')

                fig = px.bar(only_dates, x="just_date", y="dates_count", barmode='relative',color = colors, color_discrete_map="identity")
                fig.update_layout(
                    title = f'Communication {checklist_value}',
                    xaxis_title="Dates",
                    yaxis_title="Number of emails" , #  paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
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
def update_subject(people, tod, checklist, comm, engg, exect, facil, it, security):
    """ This function used for creating and updating the subject table by applying the above input
    as filters
    """

    ctx = dash.callback_context
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]
    #join dropdown box values
    join_list = engg + exect + facil + it + security
    if checklist == "Communication between different departments":
        df = merge_original_from[merge_original_from['CurrentEmploymentType_From'] != merge_original_from['CurrentEmploymentType_To_Unlisted']]
    else:
        df = merge_original_from.copy()

    if clicked_element in ['engg', 'exect', 'facil', 'IT', 'security']:
        if join_list != []:
            df = df[df['LastName_From'].isin(join_list)]
        if tod != 'both':
            final_df = df[df['Time_of_day'] == tod]
            return getData(final_df)
        else:
            return getData(df)
    elif clicked_element == 'graph':
        if tod != 'both':
            df = df[df['Time_of_day'] == tod]
            person_from = people['points'][0]['y']
            person_to = people['points'][0]['x']
            if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
                dff = df[df['LastName_From'] == person_from]
            else:
                dff = df[df['CurrentEmploymentType_From'] == person_from]
            final_df_values = dff['CurrentEmploymentType_To_Unique'].apply(lambda employee: person_to in employee)
            final_df = dff[final_df_values]

            return getData(final_df)
        else:
            person_from = people['points'][0]['y']
            person_to = people['points'][0]['x']
            if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
                dff = df[df['LastName_From'] == person_from]
            else:
                dff = df[df['CurrentEmploymentType_From'] == person_from]
            # dff = df[df['CurrentEmploymentType_From'].str.contains(person_from)]
            final_df_values = dff['CurrentEmploymentType_To_Unique'].apply(lambda employee: person_to in employee)
            final_df = dff[final_df_values]

            return getData(final_df)
    else:
        if comm != None:
            df = df.loc[df['just_date'] == comm['points'][0]['x']]
        if people != None:
            if tod != 'both':
                df = df[df['Time_of_day'] == tod]
                person_from = people['points'][0]['y']
                person_to = people['points'][0]['x']
                if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
                    dff = df[df['LastName_From'] == person_from]
                else:
                    dff = df[df['CurrentEmploymentType_From'] == person_from]
                final_df_values = dff['CurrentEmploymentType_To_Unique'].apply(lambda employee: person_to in employee)
                final_df = dff[final_df_values]

                return getData(final_df)
            else:
                person_from = people['points'][0]['y']
                person_to = people['points'][0]['x']
                if (df['CurrentEmploymentType_From'] == person_from).eq(False).all():
                    dff = df[df['LastName_From'] == person_from]
                else:
                    dff = df[df['CurrentEmploymentType_From'] == person_from]
                # dff = df[df['CurrentEmploymentType_From'].str.contains(person_from)]
                final_df_values = dff['CurrentEmploymentType_To_Unique'].apply(lambda employee: person_to in employee)
                final_df = dff[final_df_values]

                return getData(final_df)
        else:
            if tod != 'both':
                return getData(df[df['Time_of_day'] == tod])
            else:
                return getData(df)

@app.callback(
    [Output("sentiment-graph", "figure"),
     Output("articles_wc", "src"),
     Output("article-container", "children")],
    [Input("checklist2", "value"),
     Input("sentiment-graph", "clickData"),
     Input("engg", "value"),
     Input("exect", "value"),
     Input("facil", "value"),
     Input("IT", "value"),
     Input("security", "value")]
)
def update_articles(checklist_values,clickData,engg,exect,facil,IT,security):
    """ This function used for creating and updating the sentiment graph, wordcloud and articles
     container by applying the above input as filters
    """

    #list of the names of employees filtered by dropdown
    joined_list = engg + exect + facil + IT + security

    lowercase_list = list(map(lambda x: x.lower(), joined_list))

    lst = []
    for element in lowercase_list:
        l = re.split('[-. ]', element)
        lst += l
    lowercase_list = lst

    ctx = dash.callback_context
    clicked_element = ctx.triggered[0]['prop_id'].split('.')[0]

    #perform filtering and interaction based on the selected items from checklist
    if clicked_element == "checklist2":

        if checklist_values:
            articles_filtered = articles_final[articles_final['cluster'].isin(checklist_values)]
        else:
            articles_filtered = articles_final

        if lowercase_list != []:
            articles_involved = []
            for i in range(len(articles_final)):
                article_words = literal_eval(articles_final['content'].iloc[i])
                check = any(item in lowercase_list for item in article_words)
                if check:
                    articles_involved.append(articles_final['article_number'].iloc[i])
            articles_filtered = articles_filtered[articles_filtered['article_number'].isin(articles_involved)]

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
        for i in range(len(word_counts)):
            words_bag.append(word_counts[i][0])

        allWords = ' '.join([word for word in words_bag])
        wordcloud = WordCloud(width=500, height=350, max_words=60,random_state=21, max_font_size=115,
                              background_color='white',colormap='winter').generate(allWords)
        new = wordcloud.to_image()
        img2 = BytesIO()
        new.save(img2, format='PNG')
    
        text_markdown = "\t"
        with open('articles/default.txt') as this_file:
            for a in this_file.read():
                if "\n" in a:
                    text_markdown += "\n \t"
                else:
                    text_markdown += a

        return fig1,'data:image/png;base64,{}'.format(base64.b64encode(img2.getvalue()).decode()), text_markdown

    else:

        if checklist_values:

            articles_filtered = articles_final[articles_final['cluster'].isin(checklist_values)]

            if lowercase_list != []:
                articles_involved = []
                for i in range(len(articles_final)):
                    article_words = literal_eval(articles_final['content'].iloc[i])
                    check = any(item in lowercase_list for item in article_words)
                    if check:
                        articles_involved.append(articles_final['article_number'].iloc[i])
                articles_filtered = articles_filtered[articles_filtered['article_number'].isin(articles_involved)]

            if clickData is not None:

                click_article_num = clickData['points'][0]['customdata'][0]
                click_article_title = clickData['points'][0]['customdata'][1]
                articles_wc = articles_wordcloud[articles_wordcloud['article_number'] == click_article_num]

                # read and store text file
                text_markdown = "\t"
                with open('articles/' + str(click_article_num) + '.txt') as this_file:
                    for a in this_file.read():
                        if "\n" in a:
                            text_markdown += "\n \t"
                        else:
                            text_markdown += a

            else:

                article_num_list = list(articles_filtered['article_number'])
                articles_wc = articles_wordcloud[articles_wordcloud['article_number'].isin(article_num_list)]

                text_markdown = "\t"
                with open('articles/default.txt') as this_file:
                    for a in this_file.read():
                        if "\n" in a:
                            text_markdown += "\n \t"
                        else:
                            text_markdown += a

        else:

            articles_filtered = articles_final

            if lowercase_list != []:
                articles_involved = []
                for i in range(len(articles_final)):
                    article_words = literal_eval(articles_final['content'].iloc[i])
                    check = any(item in lowercase_list for item in article_words)
                    if check:
                        articles_involved.append(articles_final['article_number'].iloc[i])
                articles_filtered = articles_filtered[articles_filtered['article_number'].isin(articles_involved)]

            if clickData is not None:

                click_article_num = clickData['points'][0]['customdata'][0]
                click_article_title = clickData['points'][0]['customdata'][1]
                articles_wc = articles_wordcloud[articles_wordcloud['article_number'] == click_article_num]

                text_markdown = "\t"
                with open('articles/' + str(click_article_num) + '.txt') as this_file:
                    for a in this_file.read():
                        if "\n" in a:
                            text_markdown += "\n \t"
                        else:
                            text_markdown += a

            else:

                article_num_list = list(articles_filtered['article_number'])
                articles_wc = articles_wordcloud[articles_wordcloud['article_number'].isin(article_num_list)]

                text_markdown = "\t"
                with open('articles/default.txt') as this_file:
                    for a in this_file.read():
                        if "\n" in a:
                            text_markdown += "\n \t"
                        else:
                            text_markdown += a


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
        for i in range(len(word_counts)):
            words_bag.append(word_counts[i][0])

        allWords = ' '.join([word for word in words_bag])
        wordcloud = WordCloud(width=500, height=350, random_state=21,  max_words=60,max_font_size=115,
                              background_color='white',colormap='winter').generate(allWords)
        new = wordcloud.to_image()
        img2 = BytesIO()
        new.save(img2, format='PNG')

        return fig1,'data:image/png;base64,{}'.format(base64.b64encode(img2.getvalue()).decode()), text_markdown

if __name__ == '__main__':
    app.run_server()




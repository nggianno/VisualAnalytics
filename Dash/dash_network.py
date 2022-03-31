import pandas as pd
import networkx as nx
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dash import Dash, html, dcc, dash_table
import dash_cytoscape as cyto
import plotly.express as px
from datetime import date
from dash.dependencies import Input, Output

# Create a function to get the time of the day
def find_time_of_day(hour):
    if (hour > 4) and (hour <= 8):
        return 'Early Morning'
    elif (hour > 8) and (hour <= 12 ):
        return 'Morning'
    elif (hour > 12) and (hour <= 16):
        return'Noon'
    elif (hour > 16) and (hour <= 20) :
        return 'Evening'
    elif (hour > 20) and (hour <= 23):
        return'Night'
    elif (hour > 23) and (hour <= 1):
        return'MidNight'
    elif (x <= 4):
        return'Late Night'

def getData(df):
    return df.to_dict('rows')

# create two dataframes with the data needed
df_emails = pd.read_csv('3.Disappearance at GAStech/data/data/email headers.csv', encoding='cp1252')
df_employees = pd.read_excel('3.Disappearance at GAStech/data/data/EmployeeRecords.xlsx')

df_emails_original = df_emails.copy()

# Convert date column to date
df_emails['Date'] =pd.to_datetime(df_emails.Date)

# Find time of the day
df_emails['just_date'] = df_emails['Date'].dt.date

df_emails['day'] = df_emails['Date'].dt.day

# Create a column with the time of the day
df_emails['Time_of_day'] = df_emails['Date'].dt.hour.apply(find_time_of_day)

# # Create list with times of day
# tod = list(df_emails['Time_of_day'].unique())
tod = [{'label': 'Early Morning', 'value': 'Early Morning'},
       {'label':'Morning', 'value':'Morning'},
       {'label':'Noon', 'value':'Noon'},
       {'label':'Evening', 'value':'Evening'},
       {'label':'Night', 'value':'Night'},
       {'label':'MidNight', 'value':'MidNight'},
       {'label':'Late Night', 'value':'Late Night'}]

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

df_employees_from = df_employees.add_suffix('_From')
df_employees_to = df_employees.add_suffix('_To')

merge_from = pd.merge(df_emails, df_employees_from, how='left', left_on='From', right_on='EmailAddress_From')

final = pd.merge(merge_from, df_employees_to, how='left', left_on='To', right_on='EmailAddress_To')


# Replace no values with No_name indication
final['LastName_From'] = final['LastName_From'].fillna('CEO_diff_email')
final['LastName_To'] = final['LastName_To'].fillna('CEO_diff_email')

final['CurrentEmploymentTitle_From'] = final['CurrentEmploymentTitle_From'].fillna('CEO_diff_email')
final['CurrentEmploymentTitle_To'] = final['CurrentEmploymentTitle_To'].fillna('CEO_diff_email')

final['CurrentEmploymentType_From'] = final['CurrentEmploymentType_From'].fillna('CEO_diff_email')
final['CurrentEmploymentType_To'] = final['CurrentEmploymentType_To'].fillna('CEO_diff_email')

# final = final[final['CurrentEmploymentType_From'] != final['CurrentEmploymentType_To']]

# Add a count column with the value 1 in each line
final['count'] = 1

final['dates_count'] = final.groupby('just_date')['just_date'].transform('count')

final['combination'] = list(zip(final.From, final.To))

## Weekly analysis
week1 = final.loc[(final['just_date'] <= datetime.strptime("2014-01-10", "%Y-%m-%d").date())]
week2 = final.loc[(final['just_date'] > datetime.strptime("2014-01-10", "%Y-%m-%d").date())]
# week1['w1_combination_count'] = week1.groupby('combination')['combination'].transform('count')
# week2['w2_combination_count'] = week2.groupby('combination')['combination'].transform('count')
#
# rm_tuples = []
# for value_w1 in week1['combination']:
#     for value_w2 in week2['combination']:
#         if value_w1 == value_w2:
#             rm_tuples.append(value_w2)
#             break;
#
# final = final[~final['combination'].isin(rm_tuples)]

rm_tuples_w1 = []
for value_w1 in week1['combination']:
    for value_w2 in week2['combination']:
        if value_w1 == value_w2:
            rm_tuples_w1.append(value_w1)
            break;

only_week_1 = final[~final['combination'].isin(rm_tuples_w1)]

rm_tuples_w2 = []
for value_w2 in week2['combination']:
    for value_w1 in week1['combination']:
        if value_w2 == value_w1:
            rm_tuples_w2.append(value_w2)
            break;

only_week_2 = final[~final['combination'].isin(rm_tuples_w2)]
# dates = final['just_dates']
# dates['count'] = dates['just_dates'].values

## Pivot names
graphdf_names = final[['CurrentEmploymentType_From','CurrentEmploymentType_To','count']]
pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From', columns='CurrentEmploymentType_To',
               aggfunc='count')
# graphdf_names = final[['LastName_From','LastName_To','count']]
# pivot = pd.pivot_table(graphdf_names, values='count', index='LastName_From', columns='LastName_To',
#                aggfunc='count')
# graphdf_names = final[['CurrentEmploymentTitle_From', 'CurrentEmploymentTitle_To', 'count']]
# pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentTitle_From',
#                        columns='CurrentEmploymentTitle_To',
#                        aggfunc='count')

# Fill NaN columns with 0
pivot = pivot.fillna(0)

# pivot["total_count"] = pivot.sum(axis=1)
app = Dash(__name__)

fig = px.imshow(pivot, width = 600, height = 600,color_continuous_scale='gray_r')
# fig.update_layout(yaxis_nticks=54,xaxis_nticks=54)

dates = [date(2014, 1, 6),date(2014, 1, 7),date(2014, 1, 8),date(2014, 1, 9),date(2014, 1, 10),
         date(2014, 1, 13),date(2014, 1, 14),date(2014, 1, 15),date(2014, 1, 16),date(2014, 1, 17)]
fig2 = px.line(final, x="just_date", y="dates_count", title='Communication')

days = [6,7,8,9,10,13,14,15,16,17]
app.layout =\
    html.Div([
    html.H4('Emails'),
    dcc.Graph(id='graph1',figure=fig, clickData=None),
    dcc.Dropdown(id='dropdown', options=tod,value=None),
    dcc.Checklist(
        id = 'checklist',
        options =[
        {'label': 'Communication only week 1', 'value': 'Communication only week 1'},
        {'label': 'Communication only week 2', 'value': 'Communication only week 2'}],
        value=[]
    ),
    dash_table.DataTable(id='subject',columns=[{'name': i, 'id': i} for i in df_emails.columns],data=getData(df_emails)),
    dcc.Graph(id='graph3'),
    dcc.Graph(id='graph2',figure=fig),
        html.Label("Choose a date range for 1"),
        dcc.RangeSlider(
            id="date1",
            min=6,
            max=17,
            value=[6, 17],
            marks={str(i): {'label': str(i), 'style': {'color': 'white'}} for i in days},
        ),
    html.Label("Choose a date range for 2"),
        dcc.RangeSlider(
            id="date2",
            min=6,
            max=17,
            value=[6, 17],
            marks={str(i): {'label': str(i), 'style': {'color': 'white'}} for i in days},
        )
    ])


@app.callback(
    Output("graph1", "figure"),
    Input("date1", "value"),
    Input("dropdown", "value"),
    Input("checklist","value"))
def filter_heatmap(day, tod, checklist):
    df = final[final['day'].between(day[0], day[1])]
    # df = final[final['day'].between(day[0], day[1])]
    if tod != None:
        df = df[df['Time_of_day'] == tod]
        graphdf_names = df[['CurrentEmploymentType_From', 'CurrentEmploymentType_To', 'count']]
        pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From',
                               columns='CurrentEmploymentType_To',
                               aggfunc='count')
        # graphdf_names = df[['LastName_From', 'LastName_To', 'count']]
        # pivot = pd.pivot_table(graphdf_names, values='count', index='LastName_From',
        #                        columns='LastName_To',
        #                        aggfunc='count')
        # df = final[final['day'].between(day[0], day[1])]
        # graphdf_names = df[['CurrentEmploymentTitle_From', 'CurrentEmploymentTitle_To', 'count']]
        # pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentTitle_From',
        #                        columns='CurrentEmploymentTitle_To',
        #                        aggfunc='count')

        # Fill NaN columns with 0
        pivot = pivot.fillna(0)
        fig = px.imshow(pivot, width=600, height=600, color_continuous_scale='gray_r')
        return fig
    if checklist != []:
        print(checklist)
        if checklist == ["Communication only week 1"]:
            df = only_week_1.copy()
        elif checklist == ["Communication only week 2"]:
            print("Got in")
            df = only_week_2.copy()
        else:
            df = final.copy()
        # graphdf_names = df[['LastName_From', 'LastName_To', 'count']]
        # pivot = pd.pivot_table(graphdf_names, values='count', index='LastName_From',
        #                        columns='LastName_To',
        #                        aggfunc='count')
        graphdf_names = df[['CurrentEmploymentType_From', 'CurrentEmploymentType_To', 'count']]
        pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From',
                               columns='CurrentEmploymentType_To',
                               aggfunc='count')
        # df = final[final['day'].between(day[0], day[1])]
        # graphdf_names = df[['CurrentEmploymentTitle_From', 'CurrentEmploymentTitle_To', 'count']]
        # pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentTitle_From',
        #                        columns='CurrentEmploymentTitle_To',
        #                        aggfunc='count')

        # Fill NaN columns with 0
        pivot = pivot.fillna(0)
        fig = px.imshow(pivot, width=600, height=600, color_continuous_scale='gray_r')
        return fig
    # graphdf_names = df[['LastName_From', 'LastName_To', 'count']]
    # pivot = pd.pivot_table(graphdf_names, values='count', index='LastName_From',
    #                        columns='LastName_To',
    #                        aggfunc='count')
    graphdf_names = df[['CurrentEmploymentType_From', 'CurrentEmploymentType_To', 'count']]
    pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From',
                           columns='CurrentEmploymentType_To',
                           aggfunc='count')
    # df = final[final['day'].between(day[0], day[1])]
    # graphdf_names = df[['CurrentEmploymentTitle_From', 'CurrentEmploymentTitle_To', 'count']]
    # pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentTitle_From',
    #                        columns='CurrentEmploymentTitle_To',
    #                        aggfunc='count')

    # Fill NaN columns with 0
    pivot = pivot.fillna(0)
    fig = px.imshow(pivot, width = 600, height = 600, color_continuous_scale='gray_r')
    return fig

@app.callback(
    Output("graph2", "figure"),
    Input("date2", "value"))
def filter_heatmap(day):
    df = final[final['day'].between(day[0], day[1])]
    graphdf_names = df[['CurrentEmploymentType_From', 'CurrentEmploymentType_To', 'count']]
    pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From',
                           columns='CurrentEmploymentType_To',
                           aggfunc='count')
    # df = final[final['day'].between(day[0], day[1])]
    # graphdf_names = df[['LastName_From', 'LastName_To', 'count']]
    # pivot = pd.pivot_table(graphdf_names, values='count', index='LastName_From',
    #                        columns='LastName_To',
    #                        aggfunc='count')
    # df = final[final['day'].between(day[0], day[1])]
    # graphdf_names = df[['CurrentEmploymentTitle_From', 'CurrentEmploymentTitle_To', 'count']]
    # pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentTitle_From',
    #                        columns='CurrentEmploymentTitle_To',
    #                        aggfunc='count')

    # Fill NaN columns with 0
    pivot = pivot.fillna(0)
    fig = px.imshow(pivot, width = 600, height = 600, color_continuous_scale='gray_r')
    return fig

@app.callback(
    Output("graph3", "figure"),
    Input("graph1", "clickData"))
def filter_heatmap(people):
    if people == None:
        fig = px.line(final, x="just_date", y="dates_count", title='Communication')
        return fig
    else:
        person_from = people['points'][0]['y']
        person_to = people['points'][0]['x']
        dff = final[final['CurrentEmploymentType_From'] == person_from]
        final_df = dff[final['CurrentEmploymentType_To'] == person_to]

        final_df['dates_count'] = final_df.groupby('just_date')['just_date'].transform('count')

        fig = px.line(final_df, x="just_date", y="dates_count", title=f'From: {person_from} To {person_to}')

        return fig

@app.callback(
    Output("subject", "data"),
    Input("graph1", "clickData"))
def filter_heatmap(people):
    if people != None:
        person_from = people['points'][0]['y']
        person_to = people['points'][0]['x']
        dff = final[final['CurrentEmploymentType_From'].str.contains(person_from)]
        final_df = dff[dff['CurrentEmploymentType_To'].str.contains(person_to)]

        return getData(final_df)
    else:
        return getData(df_emails)
app.run_server(debug=True)
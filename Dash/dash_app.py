import pandas as pd
from datetime import datetime
from dash import Dash, html, dcc, dash_table
import plotly.express as px
from datetime import date
from dash.dependencies import Input, Output

### DATA PREPROCESSING ###

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

# Function needed in order to update the values in the table
def getData(df):
    return df.to_dict('rows')

# create two dataframes with the data needed
df_emails_original = pd.read_csv('3.Disappearance at GAStech/data/data/email headers.csv', encoding='cp1252')
df_employees = pd.read_excel('3.Disappearance at GAStech/data/data/EmployeeRecords.xlsx')

# Convert date column to date
df_emails_original['Date'] =pd.to_datetime(df_emails_original.Date)

# Find time of the day
df_emails_original['just_date'] = df_emails_original['Date'].dt.date

df_emails_original['day'] = df_emails_original['Date'].dt.day

# Create a column with the time of the day
df_emails_original['Time_of_day'] = df_emails_original['Date'].dt.hour.apply(find_time_of_day)

# Keep the original data to create the subject table
df_emails = df_emails_original.copy()

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

# Add a suffix that indicates if the person is the sender or the receiver
df_employees_from = df_employees.add_suffix('_From')
df_employees_to = df_employees.add_suffix('_To')

# Merge original data with From
merge_original_from = pd.merge(df_emails_original, df_employees_from, how='left', left_on='From', right_on='EmailAddress_From')

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
graphdf_names = final[['CurrentEmploymentType_From','CurrentEmploymentType_To','count']]
pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From', columns='CurrentEmploymentType_To',
               aggfunc='count')

# Fill NaN columns with 0
pivot = pivot.fillna(0)

fig = px.imshow(pivot, width = 600, height = 600,color_continuous_scale='gray_r')
# fig.update_layout(yaxis_nticks=54,xaxis_nticks=54)

# Create a list with the days
days = [6,7,8,9,10,13,14,15,16,17]

### DASH IMPLEMENTATION ###

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# Create the graphs
app.layout = \
    html.Div([
    html.Div([
    dcc.Graph(id='graph',figure=fig, clickData=None),
    dcc.Checklist(
            id = 'checklist',
            options =[
            {'label': 'Communication between all departments', 'value': 'Communication between all departments'},
            {'label': 'Communication between different departments', 'value': 'Communication between different departments'}],
            value=[]
        ),
    dcc.Dropdown(id='dropdown', options=tod,value=None)
    ],className='row'),
    html.Div([
    dash_table.DataTable(id='subject',columns=[{'name': i, 'id': i} for i in df_emails.columns],data=getData(df_emails)),
    dcc.Graph(id='communication'),
        html.Label("Choose a date range for 1"),
        dcc.RangeSlider(
            id="date",
            min=6,
            max=17,
            value=[6, 17],
            marks={str(i): {'label': str(i), 'style': {'color': 'white'}} for i in days},
        )
    ])
    ])

@app.callback(
    Output("graph", "figure"),
    Input("date", "value"),
    Input("dropdown", "value"),
    Input("checklist","value"))
def filter_heatmap(day, tod, checklist):
    if checklist == ["Communication between different departments"]:
        df = final[final['CurrentEmploymentType_From'] != final['CurrentEmploymentType_To']]
        df = df[df['day'].between(day[0], day[1])]
    else:
        df = final[final['day'].between(day[0], day[1])]
    if tod != None:
        df = df[df['Time_of_day'] == tod]

    graphdf_names = df[['CurrentEmploymentType_From', 'CurrentEmploymentType_To', 'count']]
    pivot = pd.pivot_table(graphdf_names, values='count', index='CurrentEmploymentType_From',
                           columns='CurrentEmploymentType_To',
                           aggfunc='count')

    # Fill NaN columns with 0
    pivot = pivot.fillna(0)
    fig = px.imshow(pivot, width=600, height=600, color_continuous_scale='gray_r')
    return fig

@app.callback(
    Output("communication", "figure"),
    Input("graph", "clickData"))
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

        fig = px.line(final_df, x="just_date", y="dates_count", title=f'Communication: From: {person_from} To {person_to}')

        return fig

@app.callback(
    Output("subject", "data"),
    Input("graph", "clickData"),
    Input("date", "value"),
    Input("dropdown", "value"),
    Input("checklist","value"))
def filter_heatmap(people,day,tod,checklist):
    df = merge_original_from[merge_original_from['day'].between(day[0], day[1])]
    if people != None:
        if tod != None:
            print(people)
            person_from = people['points'][0]['y']
            person_to = people['points'][0]['x']
            print(person_from)
            print(person_to)
            dff = merge_original_from[merge_original_from['CurrentEmploymentType_From'].str.contains(person_from)]
            final_df = dff[dff['CurrentEmploymentType_To'].str.contains(person_to)]

            list_of_emails_from = final_df['From'].values
            list_of_emails_to = final_df['To'].values

            df = df[df['Time_of_day'] == tod]
            df = df[df['From'].isin(list_of_emails_from)]
            final_df = df['To'].apply(lambda x: any([k in x for k in list_of_emails_to]))

            return getData(final_df)
        else:
            person_from = people['points'][0]['y']
            person_to = people['points'][0]['x']
            dff = final[final['CurrentEmploymentType_From'].str.contains(person_from)]
            final_df = dff[dff['CurrentEmploymentType_To'].str.contains(person_to)]

            list_of_emails_from = final_df['From'].values
            list_of_emails_to = final_df['To'].values

            final_df = df[df['From'].isin(list_of_emails_from)]

            # final_df = df['To'].apply(lambda x: any([k in x for k in list_of_emails_to]))

            return getData(final_df)
    else:
        if tod != None:
            return getData(df_emails_original[df_emails_original['Time_of_day'] == tod])
        else:
            return getData(df_emails_original)

app.run_server(debug=True)
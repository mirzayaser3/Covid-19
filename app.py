import dash_table
import plotly.express as px
import numpy as np
import json 
from dash.dependencies import Input, Output
import dash
import dash_core_components as dcc
import dash_html_components as html
from plotly import graph_objs as go
from plotly.graph_objs import *
from scipy.integrate import odeint
import requests
import pandas as pd
import datetime
from scipy.optimize import curve_fit

infection = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
infection = infection.drop(["Province/State", "Lat", "Long"], axis =1)
infection = infection.groupby(["Country/Region"]).sum()

recovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
recovered = recovered.drop(["Province/State","Lat", "Long"], axis=1)
recovered = recovered.groupby(["Country/Region"]).sum()

deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
deaths = deaths.drop(["Province/State","Lat", "Long"], axis=1)
deaths = deaths.groupby(["Country/Region"]).sum()

data_frame = pd.DataFrame([infection[infection.columns[-1]], recovered[recovered.columns[-1]], deaths[deaths.columns[-1]]])
data_frame = data_frame.T
data_frame.columns = ['infections','recover','dead']
data_frame['active'] = data_frame['infections'] - data_frame['recover'] - data_frame['dead']
world_data = data_frame

dates = np.array('2020-01-22', dtype=np.datetime64) + np.arange(len(infection.columns))

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.Table([
            html.Tr([
                html.Td([
            html.H4('Country'),
            dcc.Dropdown(id = 'country',
        options=[{'label': each , 'value': each} for each in world_data.index.values ],
        value="India",
                )]),
                html.Td(
                    [   
                        html.Div(id = 'present')
                    ]
                )
            ])
        ], style = {'width' : '100%', 'textAlign' : 'center'}),


        html.Div(id = "figures"),

        html.H4('SIR', style ={'textAlign' : 'center'}),

        html.Div(
        dcc.Dropdown(id = 'sir_country',
        options=[{'label': each , 'value': each} for each in world_data.index.values ],
        value="India")),

        html.Div( dcc.Graph(id = 'sir_figure')),

    ]
)

@app.callback([Output('present', 'children'), Output('figures', 'children')],
[Input('country', 'value')])

def present_data(country):

    infections, recover, dead, active = world_data.loc[f'{country}'].values
    
    layout_1 = [html.H4('PRESENT'),
              html.P([f'infections : {infections}']),
              html.P([f'recover : {recover}']),
              html.P([f'death : {dead}']),
              html.P([f'active : {active}'])]

    confirmed_figure = go.Figure()
    confirmed_figure.add_traces( go.Scatter(x= dates, y = infection.loc[f'{country}'].values, mode='lines+markers'))
    confirmed_figure.update_layout(
         height = 700,
        hovermode='x unified',
        xaxis_title="Date",
    yaxis_title="infection"
        )
    
    recovered_figure = go.Figure()
    recovered_figure.add_traces( go.Scatter(x= dates, y = recovered.loc[f'{country}'].values, mode='lines+markers'))
    recovered_figure.update_layout(
         height = 700,
        hovermode='x unified',
        xaxis_title="Date",
    yaxis_title="recover"
        )
    
    death_figure = go.Figure()
    death_figure.add_traces( go.Scatter(x= dates, y = deaths.loc[f'{country}'].values, mode='lines+markers'))
    death_figure.update_layout(
         height = 700,
        hovermode='x unified',
        xaxis_title="Date",
    yaxis_title="death"
        )

    layout_2 = [
        html.H4('Infections', style = {'textAlign' : 'center'}),
        html.Div(dcc.Graph(figure = confirmed_figure)),
         html.H4('Recoveries', style = {'textAlign' : 'center'}),
        html.Div(dcc.Graph(figure = recovered_figure)),
         html.H4('Deaths', style = {'textAlign' : 'center'}),
        html.Div(dcc.Graph(figure = death_figure )),
    ] 

    return layout_1, layout_2

@app.callback([Output('sir_figure', 'figure')],
[Input('sir_country', 'value')])

def sir_figure(country):

    length = 30 # duration for simulations 

    confirmed_data = infection.loc[f'{country}'].values

    recovered_data = recovered.loc[f'{country}'].values

    confirmed_data = confirmed_data[-30 : ]

    recovered_data = recovered_data[-30 : ]

    pre_dates = dates[-30: ]

    N = 1000000
    I_0 = confirmed_data[0]
    R_0 = recovered_data[0]
    S_0 = N - R_0 - I_0

    def SIR(y, t, beta, gamma):    
        S = y[0]
        I = y[1]
        R = y[2]
        return -beta*S*I/N, (beta*S*I)/N-(gamma*I), gamma*I

    def fit_odeint(t,beta, gamma):
        return odeint(SIR,(S_0,I_0,R_0), t, args = (beta,gamma))[:,1]

    t = np.arange(len(confirmed_data))
    params, cerr = curve_fit(fit_odeint,t, confirmed_data)
    beta,gamma = params
    prediction = list(fit_odeint(t,beta,gamma))


    fig = go.Figure()
    fig.add_trace(go.Scatter(x= pre_dates, y= prediction,
                        mode='lines+markers',
                        name='Simulated'))
    fig.add_bar(x = pre_dates, y= confirmed_data, name = "Actual")
    fig.update_layout(height = 700)

    return [fig]


if __name__ == '__main__':
    app.run_server(debug=True)



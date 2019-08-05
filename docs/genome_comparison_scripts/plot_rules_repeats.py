import dash, pandas as pd, plotly.graph_objs as go, plotly.offline as py, dash_core_components as dcc, dash_html_components as html, numpy as np, scipy.sparse as sps, sys
from sklearn.preprocessing import LabelEncoder
from collections import Counter, OrderedDict
import operator
repeat_rule_csv = '/Users/JoshuaLevy/Documents/Repositories/BitBucket/jgi-polycracker/dash_data/distachyon/dash_repeat_kmer_content.csv'

df = pd.read_csv(repeat_rule_csv)
print df
""",Subclass,Repeat,iteration,xm,length,xm_length,rule,kmer"""

df_cross = pd.crosstab(df['Repeat'],df['rule'])
df_cross = df_cross.div(df_cross.sum(axis=1),axis=0)
#print df_cross
rules = set(df['rule'].unique()) - {'rule_-1'} # FIXME figure out what to do with rules greater than or equal to 10
repeats_counter = OrderedDict(sorted(Counter(df['Repeat']).items(),key=operator.itemgetter(1)))#df['Repeat'].unique().tolist()
#print repeats_counter
repeats = repeats_counter.keys()[::-1]
subclasses = df['Subclass'].unique().tolist()
repeat_list = list(df_cross.index)
#print repeat_list
app = dash.Dash()

app.layout = html.Div([
        dcc.Dropdown(id='repeats',options= [dict(label=repeat,value=repeat) for repeat in repeats],value=repeats[0]), # [dict(label='all',value='all')] +
        dcc.Dropdown(id='iterations',options=[dict(label='all',value='all')]),# fixme update iterations with repeats chosen
        dcc.Dropdown(id='rules',options = [dict(label=rule,value=int(rule.split('_')[1])) for rule in sorted(rules)],value=0),#,min=0,max=len(rules),value=0,step=None,marks = {int(rule.split('_')[1]):rule for rule in rules}),
        dcc.Graph(id='kmer_repeat_histogram'), # FIXME histogram of kmers of particular rule
        dcc.Dropdown(id='subclass',options=[dict(label=subclass,value=subclass) for subclass in subclasses]),
        dcc.Dropdown(id='show_selected',options=[dict(label=switch,value=i) for i,switch in enumerate(['off','on'])],value=0),
        dcc.Graph(id='repeat_subclass_plot')#,
        #html.Div(id='K-mer Prevalence')
        ])

@app.callback(dash.dependencies.Output('repeat_subclass_plot','figure'),[dash.dependencies.Input('subclass','value'),dash.dependencies.Input('repeat_subclass_plot','selectedData')])
def update_repeat_plot(subclass,*selectedData):

        if subclass == 'all':
               dff_cross = df_cross
        else:
               dff_cross = df_cross[np.vectorize(lambda x: x.endswith(subclass))(repeat_list)]
        selectedpoints = np.arange(dff_cross.shape[0])
        for i, selected_data in enumerate(selectedData):
            if selected_data is not None:
                selected_index = [p['pointIndex'] for p in selected_data['points']]
                if len(selected_index) > 0:
                    selectedpoints = np.intersect1d(selectedpoints,selected_index)
        print selectedpoints
        plots = []
        for rule in rules:
            plt = go.Box(y=dff_cross[rule],name=rule,selectedpoints=selectedpoints,boxpoints='all')
            plt.unselected.marker.opacity = 0.1
            plots.append(plt)

        return dict(data=plots) #FIXME select points and output list of specific repeats

@app.callback(dash.dependencies.Output('kmer_repeat_histogram','figure'),[dash.dependencies.Input('repeats','value'),dash.dependencies.Input('rules','value'),dash.dependencies.Input('iterations','value'),dash.dependencies.Input('subclass','value'),dash.dependencies.Input('show_selected','value'),dash.dependencies.Input('repeat_subclass_plot','selectedData')])
def update_histogram(repeat,rule,iteration,subclass,selected,*selectedData):

        #print repeat, rule, iteration
        #if repeat == 'all':
        #        dff = df[df['rule'] == rule]
        #else:
        if subclass == 'all':
               dff_cross = df_cross
        else:
               dff_cross = df_cross[np.vectorize(lambda x: x.endswith(subclass))(repeat_list)]
        #print dff_cross

        selected_repeats = []
        if selected:
                selectedpoints = np.arange(dff_cross.shape[0])
                for i, selected_data in enumerate(selectedData):
                    if selected_data is not None:
                        selected_index = [p['pointIndex'] for p in selected_data['points']]
                        if len(selected_index) > 0:
                            selectedpoints = np.intersect1d(selectedpoints,selected_index)
                            selected_repeats = np.array(dff_cross.index)[selectedpoints].tolist()
                if selected_repeats:
                        dff = df[df['Repeat'].isin(selected_repeats)]
        else:
                rule = 'rule_%d'%rule
                dff = df[np.logical_and(df['Repeat'].as_matrix() == repeat,df['rule'].as_matrix() == rule)]
                #print dff
                if iteration != 'all':
                        dff = dff[dff['iteration'].as_matrix() == iteration]
                #print dff
        #print dff
        plots = []
        for kmer in dff['kmer'].unique().tolist():
                dfff = dff[dff['kmer'] == kmer]
                plots.append(go.Histogram(x=dfff['xm_length'],name='%s:%d'%(kmer,dfff.shape[0])))
        return dict(data=plots,layout=go.Layout(barmode='stack')) # FIXME this may cause it to crash

@app.callback(dash.dependencies.Output('repeats','options'),[dash.dependencies.Input('show_selected','value')])
def update_repeat_selected(selected):
        if selected:
                return [dict(label='selected',value='selected')]
        else:
                return [dict(label=repeat,value=repeat) for repeat in repeats]

@app.callback(dash.dependencies.Output('rules','options'),[dash.dependencies.Input('show_selected','value')])
def update_repeat_selected(selected):
        if selected:
                return [dict(label='selected',value='selected')]
        else:
                return [dict(label=rule,value=int(rule.split('_')[1])) for rule in sorted(rules)]

@app.callback(dash.dependencies.Output('iterations','options'),[dash.dependencies.Input('show_selected','value'),dash.dependencies.Input('repeats','value')])
def update_iteration_drop_down(selected,repeat):
        if selected:
                return [dict(label='selected',value='selected')]
        else:
                return [dict(label='all',value='all')] + [dict(label=iteration,value=iteration) for iteration in sorted(df[df['Repeat'] == repeat]['iteration'].unique().tolist())]



if __name__ == '__main__':
        app.run_server()
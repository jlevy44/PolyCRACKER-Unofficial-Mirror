import dash, pandas as pd, plotly.graph_objs as go, plotly.offline as py, dash_core_components as dcc, dash_html_components as html, numpy as np, scipy.sparse as sps, sys
from sklearn.preprocessing import LabelEncoder
from collections import Counter, OrderedDict
import operator
from sklearn.manifold import MDS


main_folder = '/Users/JoshuaLevy/Desktop/hybridum_research/dash_data_test/new/' #fixme old/
pca_input_df = main_folder+'pca_output.csv' # ['labels','x','y','z']
generations_df = main_folder+'generations.csv' #['generation','individual','parameters','score','min_cluster_size', 'min_samples', 'cluster_selection_method']
all_labels_df = main_folder+'db_labels.csv' #[parameter_names...]
dist_mat_df = main_folder+'distance_mat.csv' #[param_names by param_names]

pca_input_df = pd.read_csv(pca_input_df) # no index col
generations_df = pd.read_csv(generations_df) # no index col
all_labels_df = pd.read_csv(all_labels_df) #, index_col = 'Unnamed: 0'
dist_mat_df = pd.read_csv(dist_mat_df, index_col = 'Unnamed: 0')


indivs = generations_df['parameters'].as_matrix()
dist_mat_df=dist_mat_df.reindex(columns=indivs[::-1])
generations = generations_df['generation'].unique()
indiv_gen_dict = dict(zip(generations_df['parameters'].as_matrix().tolist(),generations_df['generation'].astype(int).as_matrix().tolist()))
indiv_scores = dict(zip(generations_df['parameters'].as_matrix().tolist(),generations_df['score'].astype(float).as_matrix().tolist()))
inds = np.vectorize(lambda x: int(x[:x.find('_')]))(indivs)
gens_all = np.vectorize(lambda x: indiv_gen_dict[x])(indivs)
scores = np.vectorize(lambda x: indiv_scores[x])(indivs)
le = LabelEncoder()
le.fit(generations_df['cluster_selection_method'])
cluster_methods = le.transform(generations_df['cluster_selection_method'])

app = dash.Dash()

app.layout = html.Div([
        dcc.Dropdown(id='indiv',options= [dict(label=indiv,value=indiv) for indiv in indivs],value=indivs[0]), # [dict(label='all',value='all')] +
        dcc.Dropdown(id='generation',options=[dict(label=generation,value=generation) for generation in ['all','selected'] + generations.tolist()],value='all'),# fixme update iterations with repeats chosen
        dcc.Dropdown(id='show_selected',options=[dict(label=switch,value=i) for i,switch in enumerate(['off','on'])],value=0),
        dcc.Graph(id='pca_plot'),
        dcc.Graph(id='distance_mat'),
        dcc.Graph(id='mds_plot'),
        dcc.Graph(id='indiv_score_plot'),
        dcc.Graph(id='parameter_space')
        ])

@app.callback(dash.dependencies.Output('pca_plot','figure'),[dash.dependencies.Input('indiv','value')])
def update_pca_plot(indiv):
    plots = []
    labels = all_labels_df[indiv].unique()
    for label in labels:
        dff = pca_input_df[all_labels_df[indiv]==label]
        plots.append(go.Scatter3d(x=dff['x'],y=dff['y'],z=dff['z'],text=dff['labels'],mode='markers',marker=dict(size=2),name='Cluster %d'%label))
    return dict(data=plots)

@app.callback(dash.dependencies.Output('distance_mat','figure'),[dash.dependencies.Input('generation','value'),dash.dependencies.Input('indiv_score_plot','selectedData')])
def update_dist_matrix(generation,*selectedData): # fixme rotate distance matrix
    if generation == 'all':
        return dict(data=[go.Heatmap(x=indivs,y=indivs[::-1],z=dist_mat_df.as_matrix())]) #[::-1]
    elif generation == 'selected':
        selectedpoints = np.arange(indivs.shape[0])
        for i, selected_data in enumerate(selectedData):
            if selected_data is not None:
                selected_index = [p['pointIndex'] for p in selected_data['points']]
                if len(selected_index) > 0:
                    selectedpoints = np.intersect1d(selectedpoints,selected_index)
        return dict(data=[go.Heatmap(x=indivs,y=indivs[::-1],z=dist_mat_df.as_matrix(),selectedpoints=selectedpoints)])#[::-1]
    else:
        vals = indivs[generations_df['generation']==generation]
        dff = dist_mat_df.loc[vals,vals[::-1]]#[::-1] fixme do columns instead
        return dict(data=[go.Heatmap(x=vals,y=vals[::-1],z=dff.as_matrix())])#[::-1]

@app.callback(dash.dependencies.Output('mds_plot','figure'),[dash.dependencies.Input('generation','value'),dash.dependencies.Input('indiv_score_plot','selectedData')])
def update_MDS(generation,*selectedData):
    mds = MDS(n_components=3,dissimilarity='precomputed')
    if generation == 'all':
        plots = []
        t_data = mds.fit_transform(dist_mat_df.loc[indivs,indivs])
        for gen in generations:
            plots.append(go.Scatter3d(x=t_data[generations_df['generation']==gen,0],y=t_data[generations_df['generation']==gen,1],z=t_data[generations_df['generation']==gen,2],mode='markers',name='Generation %d'%gen,text=generations_df['parameters'][generations_df['generation']==gen]))
        return dict(data=plots)
    elif generation == 'selected':
        selectedpoints = np.arange(indivs.shape[0])
        for i, selected_data in enumerate(selectedData):
            if selected_data is not None:
                selected_index = [p['pointIndex'] for p in selected_data['points']]
                if len(selected_index) > 0:
                    selectedpoints = np.intersect1d(selectedpoints,selected_index)
        vals = indivs[selectedpoints]
        gens = np.vectorize(lambda x: indiv_gen_dict[x])(vals)
        t_data = mds.fit_transform(dist_mat_df.loc[vals,vals])
        plots = []
        for gen in np.unique(gens).tolist():
             plots.append(go.Scatter3d(x=t_data[gens == gen,0],y=t_data[gens == gen,1],z=t_data[gens == gen,2],name='Generation %d'%gen,mode='markers',text=vals[gens==gen]))
        return dict(data=plots)
    else:
        vals = indivs[generations_df['generation']==generation]
        t_data = mds.fit_transform(dist_mat_df.loc[vals,vals])
        return dict(data=[go.Scatter3d(x=t_data[:,0],y=t_data[:,1],z=t_data[:,2],name='Generation %d'%generation,mode='markers',text=vals)])

@app.callback(dash.dependencies.Output('indiv_score_plot','figure'),[dash.dependencies.Input('indiv_score_plot','selectedData')])
def update_indiv_score(*selectedData):

    selectedpoints = np.arange(indivs.shape[0])
    for i, selected_data in enumerate(selectedData):
        if selected_data is not None:
            selected_index = [p['pointIndex'] for p in selected_data['points']]
            if len(selected_index) > 0:
                selectedpoints = np.intersect1d(selectedpoints,selected_index)

    plots = []
    for gen in np.unique(gens_all).tolist():
        plt = go.Scatter(x=inds[gens_all==gen],y=scores[gens_all==gen],name= 'Generation %d'%gen,text=indivs[gens_all==gen],selectedpoints=selectedpoints)
        plt.unselected.marker.opacity = 0.1
        plots.append(plt)
    return dict(data=plots)

@app.callback(dash.dependencies.Output('parameter_space','figure'),[dash.dependencies.Input('indiv_score_plot','selectedData')])
def update_parameter_space(*selectedData):

    selectedpoints = np.arange(indivs.shape[0])
    for i, selected_data in enumerate(selectedData):
        if selected_data is not None:
            selected_index = [p['pointIndex'] for p in selected_data['points']]
            if len(selected_index) > 0:
                selectedpoints = np.intersect1d(selectedpoints,selected_index)
    #dff = generations_df.loc[selectedpoints,:]
    #CM = cluster_methods[selectedpoints]
    print cluster_methods, generations_df['min_cluster_size'], generations_df['min_samples'], scores
    plt = go.Scatter3d(x=cluster_methods,y=generations_df['min_cluster_size'].as_matrix(),z=generations_df['min_samples'].as_matrix(),mode='markers',selectedpoints=selectedpoints,marker=dict(color=scores,colorscale='Viridis'))
    print plt
    #plt.unselected.marker.opacity = 0.1
    return dict(data=[plt],layout=go.Layout(xaxis=dict(ticktext=le.inverse_transform(np.unique(cluster_methods)))))


if __name__ == '__main__':
        app.run_server()
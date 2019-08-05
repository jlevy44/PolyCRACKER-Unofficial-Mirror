import dash, pandas as pd, plotly.graph_objs as go, plotly.offline as py, dash_core_components as dcc, dash_html_components as html, numpy as np, scipy.sparse as sps, sys
from sklearn.preprocessing import LabelEncoder, normalize
from collections import defaultdict
import click
import dash_table_experiments as dt
import copy
#FIXME use kmers as index???, add quality check and ability to toggle between quality and rule plots, and turn on and off absolute vs relative abundance

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)


@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def plot_rules_passed():
    pass

global rule_csv,sparse_kmer_coords,sparse_kmer_data,pca_data, app

@plot_rules_passed.command()
@click.option('-r', '--rule_csv', default='./dash_data/kmer_master_matrix_rules.csv', help='Kmer matrix rules file.', show_default=True, type=click.Path(exists=False))
@click.option('-q', '--quality_csv', default='./dash_data/genome_quality_check.csv', help='Quality check kmer count matrix file.', show_default=True, type=click.Path(exists=False))
@click.option('-s', '--sparse_kmer_coords', default='./dash_data/kmer.count.coords.csv', help='Sparse kmer matrix, coordinate csv file.', show_default=True, type=click.Path(exists=False))
@click.option('-npz', '--sparse_kmer_data', default='./dash_data/kmer.count.npz', help='Npz file of sparse matrix of genomic window vs kmer count.', show_default=True, type=click.Path(exists=False))
@click.option('-pca', '--pca_data', default='./dash_data/pca_data.csv', help='PCA data to plot, csv file.', show_default=True, type=click.Path(exists=False))
def plot_rules(rule_csv,quality_csv,sparse_kmer_coords,sparse_kmer_data,pca_data):

    #rule_csv = rule_csv#'./dash_data/kmer_master_matrix_rules.csv'#kmer_master_count_chi2_matrix_rules.csv'
    #sparse_kmer_coords = sparse_kmer_coords#'./dash_data/kmer.count.coords.csv'
    #sparse_kmer_data = sparse_kmer_data#'./dash_data/kmer.count.npz'
    #pca_data = pca_data#'./dash_data/pca_data.csv'

    #try:
    #    rule_csv,sparse_kmer_coords,sparse_kmer_data,pca_data = tuple(sys.argv[1:5])
    #except:
    #    rule_csv,sparse_kmer_coords,sparse_kmer_data,pca_data = './dash_data/kmer_master_matrix_rules.csv','./dash_data/kmer.count.coords.csv','./dash_data/kmer.count.npz','./dash_data/pca_data.csv'

    app = dash.Dash()
    sparse_kmer_data = sps.load_npz(sparse_kmer_data)
    sparse_kmer_coords = pd.read_csv(sparse_kmer_coords,index_col = 'Unnamed: 0')
    #sparse_kmer_data = './sparse_kmer_count_matrix.p'
    df = pd.read_csv(rule_csv,index_col='Unnamed: 0')
    absolute_df = df.copy()
    q_df = pd.read_csv(quality_csv,index_col='Unnamed: 0')
    q_df = q_df.reindex(columns=[col for col in list(q_df) if col != 'Rules'] + ['Rules'])
    q_df_absolute = q_df.copy()
    #q_df.loc[:,[col for col in list(q_df) if col != 'Rules']] = normalize(q_df.loc[:,[col for col in list(q_df) if col != 'Rules']].as_matrix().astype(float),norm='max')
    X = q_df.iloc[:,0:-1].as_matrix()
    q_df.iloc[:,0:-1] = X / X.sum(axis=1)[:, np.newaxis].astype(float)
    pca_data = pd.read_csv(pca_data,index_col='kmers')
    #sparse_kmer_data = pd.read_pickle(sparse_kmer_data)
    X , y = df.iloc[:,0:-1].as_matrix(), df.iloc[:,-1].as_matrix()
    if 'kmer_master_count_chi2_matrix_rules.csv' in rule_csv:
        n_col = X.shape[1]
        X1 = X[:,:n_col/2]
        X1 /= X1.sum(axis=1).astype(float)[:, np.newaxis]
        X[:,:n_col/2] = X1
    else:
        X = X / X.sum(axis=1)[:, np.newaxis].astype(float)
    df.iloc[:,0:-1] = X
    df = df.drop(columns=[col for col in list(df) if '-chi2' in col])
    absolute_df = absolute_df.drop(columns=[col for col in list(df) if '-chi2' in col])

    le = LabelEncoder()
    kmers = list(df.index)
    le.fit(kmers)
    #df = df.reindex(index=range(len(df)))
    #sparse_kmer_data = sparse_kmer_data.rename(columns=dict(zip(kmers,le.transform(kmers))),axis=1)
    chromosomes = sparse_kmer_coords['chr'].unique()#sparse_kmer_data['chr'].unique()
    chromosomes_dict = dict(enumerate(sparse_kmer_coords['chr'].unique().tolist()))
    #print df.index[0:10]
    df = df.reset_index(drop=True) #FIXME change index to kmers
    absolute_df = absolute_df.reset_index(drop=True)
    q_df = q_df.reset_index(drop=True)
    q_df_absolute = q_df_absolute.reset_index(drop=True)
    unassigned_abundance_df = [[q_df,q_df_absolute],[df,absolute_df]]
    #print pca_data.index[0:10]
    pca_data = pca_data.reset_index(drop=True)#pca_data.rename(index=dict(zip(pca_data.index,le.transform(pca_data.index))))
    #print df, pca_data
    rules = ['rule %d'%i for i,rule in enumerate(set(np.unique(df['Rules'])) - {'rule -1'})]
    #print {i:rule for i,rule in enumerate(rules)}
    cols_orig = [set(q_df.columns.values) - {'Rules'} ,set(df.columns.values) - {'Rules'}] #FIXME fix styling and then test distachyon heavy and hybridum heavy analysis, maybe add more species
    cols = copy.deepcopy(cols_orig)
    #c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(cols))]
    #color_array = np.ones((df.shape[0],len(cols))).astype(str)
    #for i in range(len(cols)):
    #    color_array[:,i] = c[i]
    app.layout = html.Div([
            html.Div([
                dcc.Graph(style=dict(width='800',height='800'),id='rules_plot'),
                html.Label('Absolute/Relative Abundance'),
                dcc.Dropdown(id='abundance',options = [dict(label=switch,value=i) for i,switch in enumerate(['relative','absolute'])],value=0),
                html.Label('Scaffolded Subgenomes / Unassigned Scaffold Genomes'),
                dcc.Dropdown(id='unassigned',options = [dict(label=switch,value=i) for i,switch in enumerate(['prescaffolded','scaffolded'])],value=1),
                html.Label('Rules'),
                dcc.Dropdown(id='rules',options=[dict(label=switch,value=i) for i,switch in enumerate(rules)]),#,min=0,max=len(rules),value=0,step=None,marks = {i:rule for i,rule in enumerate(rules)}),
                #dcc.Slider(id='chromosomes',min=0,max=len(chromosomes),value=0,step=None,marks = {i:chrom for i,chrom in enumerate(chromosomes.tolist())}),
                dcc.Markdown(children='\n\n'),#,
                dt.DataTable(rows=[{'Species':species,'Order Number': i, 'Display Name':species} for i,species in enumerate(cols[0])],id='datatable',editable=True,sortable=True)#,row_selectable=True,filterable=True,editable=True,sortable=True,selected_row_indices=[],id='datatable')

            ], className="six columns"),
            html.Div([
                dcc.Graph(style=dict(width='800'),id='pca_plot'),
                html.Label('Chromosome'),
                dcc.Dropdown(id='chromosomes',options = [dict(label=chrom,value=i) for i,chrom in enumerate(chromosomes)],value=0),#{chrom:i for i,chrom in enumerate(chromosomes)},value=0),
                html.Label('Plot Kmers Selected in Prevalence Boxplots'),
                dcc.Dropdown(id='original',options = [dict(label=switch,value=i) for i,switch in enumerate(['off','on'])],value=1),
                html.Label('Intersection Mode'),
                dcc.Dropdown(id='intersection',options = [dict(label=switch,value=i) for i,switch in enumerate(['off','on'])],value=0),
                dcc.Graph(style=dict(width='800'),id='spatial_kmer_plot')


                #dcc.Graph(id='rules_plot_filter')

                ], className="six columns")
        ], style={'display': 'inline-block'})

    app.css.append_css({
        'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
    })

    @app.callback(dash.dependencies.Output('datatable', 'rows'),[dash.dependencies.Input('unassigned','value')])
    def update_datatable(unassigned):
        return [{'Species':species,'Order Number': i, 'Display Name':species+str(' ')} for i,species in enumerate(cols[unassigned])]


    @app.callback(dash.dependencies.Output('rules_plot','figure'),[dash.dependencies.Input('rules','value'),dash.dependencies.Input('intersection','value'),dash.dependencies.Input('unassigned','value'),dash.dependencies.Input('abundance','value'),dash.dependencies.Input('datatable', 'rows'),dash.dependencies.Input('rules_plot','selectedData')])
    def update_graph(rule_value,intersection,unassigned,abundance,datatable,*selectedData):
        #print(selectedData)
        datatable = pd.DataFrame(datatable).sort_values(['Order Number'])
        abundance = unassigned_abundance_df[unassigned][abundance]
        dff = abundance[abundance['Rules'] == 'rule %d'%rule_value]
        #print dff
        #colors = color_array
        selectedpoints = np.arange(dff.shape[0])
        #print selectedData
        for i, selected_data in enumerate(selectedData):
            if selected_data is not None:
                if intersection:
                    intersected_points = defaultdict(list)
                    for p in selected_data['points']:
                        intersected_points[p['x']].append(p['pointIndex'])
                    selected_index = list(reduce(lambda x,y: set(x).intersection(set(y)),intersected_points.values()))
                else:
                    selected_index = [p['pointIndex'] for p in selected_data['points']]
                if len(selected_index) > 0:
                    selectedpoints = np.intersect1d(selectedpoints,selected_index)
                    #colors[selectedpoints,:] = 'black'
                    #print selectedpoints
        #print colors

        data = []
        for col1,col2 in zip(datatable['Species'].as_matrix().tolist(),datatable['Display Name'].as_matrix().tolist()): # cols[unassigned]
            #print col
            #print dff[col]
            #print col1,col2
            plt = go.Box(y=dff[str(col1)].as_matrix(),name=str(col2),boxpoints='all',selectedpoints=selectedpoints) #{'unselected': {'marker': {'opacity': 0.2}}, 'type': 'bar'} marker=dict(color=colors[:,i]),fillcolor=c[i],
            plt.unselected.marker.opacity = 0.01
            data.append(plt)
        return dict(data=data,layout=dict(title='Relative Prevalence of Kmers Across Genomes'))

    """
    @app.callback(dash.dependencies.Output('rules_plot_filter','figure'),[dash.dependencies.Input('rules','value'),dash.dependencies.Input('rules_plot','selectedData')])
    def update_filtered_graph(rule_value,*selectedData):
        dff = df[df['Rules'] == 'rule %d'%rule_value]
        data = []
        selectedpoints = dff.index
        #print selectedpoints
        for i, selected_data in enumerate(selectedData):
            if selected_data is not None:
                #print selected_data['points']
                selected_index = [p['pointIndex'] for p in selected_data['points']]
                #print selected_index
                if len(selected_index) > 0:
                    selectedpoints = np.intersect1d(selectedpoints,selected_index)
                    #print selectedpoints
                    dff = dff.iloc[selectedpoints,:]
        for col in cols:
            data.append(go.Box(y=dff[col],name=col,boxpoints='all'))
        return dict(data=data)"""


    @app.callback(dash.dependencies.Output('pca_plot','figure'),[dash.dependencies.Input('rules','value'),dash.dependencies.Input('intersection','value'),dash.dependencies.Input('original','value'),dash.dependencies.Input('rules_plot','selectedData')])
    def update_pca_graph(rule,intersection,switch,*selectedData):
        #dff = df[df['Rules'] == 'rule %d'%rule_value]
        dff = pca_data
        selectedpoints = pca_data.index#dff.index
        if switch:
            #print selectedpoints
            #FIXME selected data defaults to all data in selected rule, should show entire plot if nothing is selected
            for i, selected_data in enumerate(selectedData):
                if selected_data is not None:
                    dff = dff[dff['rule'] == 'rule %d'%rule]
                    #print dff
                    #print selected_data['points']
                    if intersection:
                        intersected_points = defaultdict(list)
                        for p in selected_data['points']:
                            intersected_points[p['x']].append(p['pointIndex'])
                        selected_index = list(reduce(lambda x,y: set(x).intersection(set(y)),intersected_points.values()))
                    else:
                        selected_index = [p['pointIndex'] for p in selected_data['points']]
                    #print selected_index
                    #print selected_index
                    if len(selected_index) > 0:
                        selectedpoints = np.intersect1d(selectedpoints,selected_index)
                        #print selectedpoints
                        dff = dff.iloc[selectedpoints,:]# fixme
        plots = []
        #plots.append(go.Scatter3d(x=dff['x'],y=dff['y'],z=dff['z'],name=dff['rule'],mode='markers',marker=dict(size=2),text=dff['label'],selectedpoints=selectedpoints))

        for name in dff['rule'].unique():
            dfff = dff[dff['rule'].as_matrix()==name]
            plots.append(go.Scatter3d(x=dfff['x'],y=dfff['y'],z=dfff['z'],name=name,mode='markers',marker=dict(size=2),text=dfff['label'])) #selectedpoints=selectedpoints

        #for col in cols:
        #    data.append(go.Box(y=dff[col],name=col,boxpoints='all'))
        return dict(data=plots,layout=dict(title='PCA Graph kmer Prevalence'))

    @app.callback(dash.dependencies.Output('spatial_kmer_plot','figure'),[dash.dependencies.Input('chromosomes','value'),dash.dependencies.Input('intersection','value'),dash.dependencies.Input('original','value'),dash.dependencies.Input('rules_plot','selectedData')])
    def update_spatial_graph(chromosome,intersection,switch,*selectedData):
        dff = sparse_kmer_coords #sparse_kmer_data[sparse_kmer_data['chr'].as_matrix()==chromosome]
        chromosome = chromosomes_dict[chromosome]
        sparse_kmer_data_dff = sparse_kmer_data[sparse_kmer_coords['chr'].as_matrix()==chromosome,:]
        dff = dff[sparse_kmer_coords['chr'].as_matrix()==chromosome]
        dff['sum'] = sparse_kmer_data_dff.sum(axis=1)#.toarray()#dff.iloc[:,2:].sum(axis=1)
        selectedpoints = np.arange(sparse_kmer_data_dff.shape[1])
        if switch:
            for i, selected_data in enumerate(selectedData):
                if selected_data is not None:
                    #print selected_data['points']
                    if intersection:
                        intersected_points = defaultdict(list)
                        for p in selected_data['points']:
                            intersected_points[p['x']].append(p['pointIndex'])
                        selected_index = list(reduce(lambda x,y: set(x).intersection(set(y)),intersected_points.values()))
                    else:
                        selected_index = [p['pointIndex'] for p in selected_data['points']]
                    #print selected_index
                    if len(selected_index) > 0:
                        selectedpoints = np.intersect1d(selectedpoints,selected_index)
                        #print selectedpoints
                        dff['sum'] = sparse_kmer_data_dff[:,selectedpoints].sum(axis=1)#.toarray()#dff.loc[:,selectedpoints].sum(axis=1)
        plots = []
        plots.append(go.Scatter(x=dff['xi'],y=dff['sum']))
        return dict(data=plots,layout=dict(title='Distribution of Kmers Across Chromosome %s'%chromosome))


    app.run_server(debug=True)



    # FIXME features to add
    # FIXME add PCA graph
    # FIXME add spatial chromosome plot of kmers

    """
    def highlight(x,y):
        def callback(*selectedDatas):
            selectedpoints = df.index
            for i, selected_data in enumerate(selectedDatas):
                if selected_data is not None:
                    selected_index = [p['customdata'] for p in selected_data['points']]
                    if len(selected_index ) > 0:
                        selectedpoints = np.intersect1d(selectedpoints,selected_index)
            data = []
            for col in cols:
                data.append(go.Box(y=df[col]))
            return dict(data=data)"""

if __name__ == '__main__':
    plot_rules_passed()


"""
@polycracker.command()
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for plotting kmer count matrix is done.', type=click.Path(exists=False))
@click.option('-csv', '--rule_csv', default = './kmer_master_count_chi2_matrix_rules.csv', show_default = True, help='Kmer count matrix with appended rule labels. Default includes chi-squared values.', type=click.Path(exists=False))
def plot_rules(work_dir,rule_csv):

    X , y = df.iloc[:,1:-1].as_matrix(), df.iloc[:,-1].as_matrix()
    if 'kmer_master_count_chi2_matrix_rules.csv' in rule_csv:
        n_col = X.shape[1]
        X1 = X[:,:n_col/2]
        X1 /= X1.sum(axis=1).astype(float)[:, np.newaxis]
        X[:,:n_col/2] = X1
    else:
        X = X / X.sum(axis=1)[:, np.newaxis].astype(float)
    col_names = list(df)[1:-1]
    for rule in set(y) - {'rule -1'}:
        plots = []
        x = X[y == rule,:]
        for i, col in enumerate(col_names):
            plots.append(go.Box(y=x[:,i],name=col,boxpoints='all'))
        py.plot(go.Figure(data=plots),filename=work_dir+rule.replace(' ','_')+'.html',auto_open=False)
        """
#!/opt/conda/envs/pCRACKER_p27/bin/python
# All rights reserved.
from collections import Counter, defaultdict, OrderedDict
import cPickle as pickle
import errno
from itertools import combinations, permutations
import itertools
import os
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import click
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import pybedtools
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
import hdbscan
import networkx as nx
from pybedtools import BedTool
from pyfaidx import Fasta
import scipy.sparse as sps
from scipy.stats import pearsonr, chi2_contingency
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FactorAnalysis, LatentDirichletAllocation, NMF
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.metrics import calinski_harabaz_score, silhouette_score
# from evolutionary_search import maximize


RANDOM_STATE=42

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='1.1.3')
def utilities():
    pass


def create_path(path):
    """Create a path if directory does not exist, raise exception for other errors"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


@utilities.command(name='extract_scaffolds_fasta')
@click.pass_context
@click.option('-fi', '--input_fasta', default='', help='Path to genome.', show_default=True, type=click.Path(exists=False))
@click.option('-d', '--labels_dict', help='Comma delimited dictionary of text to find in scaffold to new output file head name, omitting .fa or .fasta. Example: text1:label1,text2:label2')
@click.option('-o', '--output_dir', default='./', help='Output directory.', show_default=True, type=click.Path(exists=False))
def extract_scaffolds_fasta(ctx,input_fasta,labels_dict,output_dir):
    """Extract scaffolds from fasta file using names that the sequences start with."""
    output_dir += '/'
    scaffolds = np.array(Fasta(input_fasta).keys())
    labels_dict = {text:label for text,label in [tuple(mapping.split(':')) for mapping in labels_dict.split(',')]}
    def relabel(scaffold,labels_dict = labels_dict):
        for text in labels_dict:
            if text in scaffold:
                return labels_dict[text]
        return 'unlabelled'
    labels = np.vectorize(relabel)(scaffolds)
    txt_files = []
    for label in set(labels):
        with open(output_dir+label+'.txt','w') as f:
            f.write('\n'.join(scaffolds[labels==label]))
        txt_files.append(output_dir+label+'.txt')
    ctx.invoke(txt2fasta,txt_files = ','.join(txt_files), reference_fasta = input_fasta)


def kmer_set(k,kmers):
    final_kmer_set = []
    for kmer in kmers:
        if len(kmer) == k:
            final_kmer_set.append(kmer)
        else:
            final_kmer_set.extend(set([kmer[i:i+k] for i in xrange(len(kmer) - k)]))
    return set(final_kmer_set)


@utilities.command(name='polyploid_diff_kmer_comparison')
@click.option('-p', '--analysis_path', default='./polyploid_comparison/', help='Work directory where computations for polyploid comparison is done.', type=click.Path(exists=False), show_default=True)
@click.option('-l', '--kmer_list_path', default='./polyploid_comparison/diff_kmer_lists/', help='Directory containing list of informative differential kmers for each polyploid, pickle files.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--kmer_type_path', default='./polyploid_comparison/diff_kmer_types/', help='Directory containing dictionaries of distributions of kmer class/subclass type for each polyploid, pickle files.', type=click.Path(exists=False), show_default=True)
def polyploid_diff_kmer_comparison(analysis_path, kmer_list_path, kmer_type_path):
    """Compare highly informative differential kmers between many different polyCRACKER runs/polyploids. Find intersections between them.
    In development: May be better to modify bio_hyp_class pipeline to handle this problem."""
    import venn
    # create output directory
    try:
        os.makedirs(analysis_path)
    except:
        pass
    if kmer_list_path.endswith('/') == 0:
        kmer_list_path += '/'
    if kmer_type_path.endswith('/') == 0:
        kmer_type_path += '/'
    if analysis_path.endswith('/') == 0:
        analysis_path += '/'
    polyploid_top_kmers = defaultdict(list)
    polyploid_top_subgenome_share_rate = defaultdict(list)
    for kmer_pickle in os.listdir(kmer_list_path):
        if kmer_pickle.endswith('.p'):
            polyploid_top_kmers[kmer_pickle.split('.')[0]] = pickle.load(open(kmer_list_path+kmer_pickle,'rb'))
    lowest_kmer_size = len(min(polyploid_top_kmers[min(polyploid_top_kmers.keys(), key=lambda x: len(min(polyploid_top_kmers[x], key=len)))], key = len))
    #print lowest_kmer_size
    for polyploid in polyploid_top_kmers:
        polyploid_top_subgenome_share_rate[polyploid] = 1. - len(set(polyploid_top_kmers[polyploid]))/float(len(polyploid_top_kmers[polyploid]))
        polyploid_top_kmers[polyploid] = kmer_set(lowest_kmer_size, polyploid_top_kmers[polyploid])
    py.plot(go.Figure(layout=go.Layout(title='Top Differential Kmer Share Rate Between Polyploid Subgenomes'),
                      data=[go.Bar(x=polyploid_top_subgenome_share_rate.keys(),y=polyploid_top_subgenome_share_rate.values(),name=polyploid)]),
            filename=analysis_path+'polyploid_subgenome_top_kmer_share.html', auto_open=False)
    #pickle.dump(polyploid_top_kmers,open('top_kmers.p','wb'))
    #print polyploid_top_kmers
    n_polyploids = len(polyploid_top_kmers.keys())
    labels = venn.get_labels(polyploid_top_kmers.values(), fill=['number', 'logic'])
    if n_polyploids == 2:
        fig, ax = venn.venn2(labels, names=polyploid_top_kmers.keys())
    elif n_polyploids == 3:
        fig, ax = venn.venn3(labels, names=polyploid_top_kmers.keys())
    elif n_polyploids == 4:
        fig, ax = venn.venn4(labels, names=polyploid_top_kmers.keys())
    elif n_polyploids == 5:
        fig, ax = venn.venn5(labels, names=polyploid_top_kmers.keys())
    elif n_polyploids == 6:
        fig, ax = venn.venn6(labels, names=polyploid_top_kmers.keys())
    else:
        print 'Invalid number of polyploids'
    if n_polyploids in range(2,7):
        fig.savefig(analysis_path+'polyploid_comparison_top_differential_kmers.png')
    try:
        for class_label in ['class', 'subclass']:
            polyploid_top_kmer_types = defaultdict(list)
            for kmer_pickle in os.listdir(kmer_type_path):
                if kmer_pickle.endswith(class_label+'.p'):
                    polyploid_top_kmer_types[kmer_pickle.split('.')[0]] = pickle.load(open(kmer_type_path+kmer_pickle,'rb'))
            plots = []
            for polyploid in polyploid_top_kmer_types:
                plots.append(go.Bar(x=polyploid_top_kmer_types.keys(),y=polyploid_top_kmer_types.values(),name=polyploid))
            df = pd.DataFrame(polyploid_top_kmer_types)
            df.fillna(0, inplace=True)
            df.to_csv(analysis_path+'Polyploid_Comparison_Differential_Kmers_By_%s.csv'%class_label)
            indep_test_results = chi2_contingency(df.as_matrix(), correction=True)
            fig = go.Figure(data=plots,layout=go.Layout(title='X**2=%.2f, p=%.2f'%indep_test_results[0:2],barmode='group'))
            py.plot(fig, filename=analysis_path+'Polyploid_Comparison_Differential_Kmers_By_%s.html'%class_label, auto_open=False)
    except:
        pass
    """I remember that.  I guess the questions I want to answer are:  can we define particular repeats ("full length"
    rather than just the kmers) that are marking our subgenomes?  What are they (type/class/family)?

    take repeats fasta file and bbmap it against main fasta file... Find clustering matrix and see if separates.

    then see which class TE helps separate it and find top TEs
    """


@utilities.command()
@click.pass_context
@click.option('-i','--input_dict',default='subgenome_1:./fasta_files/subgenome_1.fa,SpeciesA:./SpeciesA.fasta',show_default=True, help='MANDATORY: Comma delimited mapping of all subgenome names to their subgenome fasta files.', type=click.Path(exists=False))
@click.option('-f', '--folder_delimiter', default='./folder/,_,4', show_default=True, help='OPTIONA: Comma delimited list of folder containing genomes, delimiters from which to construct short names, and number of characters to use for short name. Save fastas as AAAA_xxx_v0.fa format. Supercedes -i option, unless default value not modified.', type=click.Path(exists=False))
@click.option('-kv', '--default_kcount_value', default = 1, help='If kmer hits in subgenome is 0, this value will replace it.', show_default = True)
@click.option('-dk', '--ratio_threshold', default=20, help='Value indicates that if the counts of a particular kmer in a subgenome divided by any of the counts in other subgenomes are greater than this value, then the kmer is differential.', show_default=True)
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for biological hypothesis testing is done.', type=click.Path(exists=False))
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
@click.option('-l', '--kmer_length', default=23, help='Length of kmers to find.', show_default=True)
@click.option('-ft', '--fast_tree', is_flag=True, help='Make quick estimated phylogenetic tree out of kmer matrix.')
@click.option('-min', '--min_count', default = 3, show_default=True, help='Minimum count for kmer counting.')
@click.option('-sk', '--skip_kcount', is_flag=True, help='Skip generation of kcount files. kcount files must be named by the following convention: subgenome_1 used in input dict -> subgenome_1.kcount')
@click.option('-max', '--max_val', default = 100, show_default=True, help='Minimum count for maximum count of kmer between subgenomes.')
def bio_hyp_class(ctx,input_dict, folder_delimiter, default_kcount_value, ratio_threshold, work_dir, blast_mem, kmer_length, fast_tree, min_count, skip_kcount, max_val):
    """Generate count matrix of kmers versus genomes. Optionally can transpose this matrix and find kmer guide tree / approximated phylogeny based on differential kmer counts. Changing settings to include all kmers can generate a more accurate guide tree."""
    #from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import LabelEncoder
    #from scipy.stats import chisquare
    if not work_dir.endswith('/'):
        work_dir += '/'
    try:
        os.makedirs(work_dir)
    except:
        pass
    blast_memStr = "export _JAVA_OPTIONS='-Xms5G -Xmx%sG'"%(blast_mem)

    if folder_delimiter != './folder/,_,4':
        folder, delimiter, n_chars = tuple(folder_delimiter.split(','))
        if n_chars == 'n':
            n_char = False
        else:
            n_chars = int(n_chars)
        fastas = [fasta for fasta in os.listdir(folder) if fasta.endswith('.fa') or fasta.endswith('.fasta')]
        short_names = map(lambda x: x[:n_chars] + x[x.find(delimiter)+1:x.rfind(delimiter)] if n_char else x.split('.')[0],fastas)
        subgenome_labels = dict(zip(short_names,map(lambda x: folder+'/'+x,fastas)))
    else:
        subgenome_labels = {subgenome_name:subgenome_fasta for subgenome_name,subgenome_fasta in sorted([tuple(mapping.split(':')) for mapping in input_dict.split(',')])}
    kmer_matrix = []
    # FIXME GENERATE MATRIX HERE
    #def return_kcounts(subgenome_name):
    #kmer_counts = pd.read_table(work_dir+subgenome_name+'.kcount',sep=None)
    #return dict(zip(kmer_counts[0],kmer_counts[1]))
    #    return dict(zip(os.popen("awk '{ print $1 }' %s"%(work_dir+subgenome_name+'.kcount')).read().splitlines(),map(int,os.popen("awk '{ print $2 }' %s"%(work_dir+subgenome_name+'.kcount')).read().splitlines())))
    return_kcounts = lambda s_name: dict(itertools.izip(os.popen("awk '{ print $1 }' %s"%(work_dir+s_name+'.kcount')).read().splitlines(),map(int,os.popen("awk '{ print $2 }' %s"%(work_dir+s_name+'.kcount')).read().splitlines())))
    for subgenome_name in subgenome_labels:
        click.echo(subgenome_name)
        if not skip_kcount:
            if kmer_length <= 31:
                subprocess.call(blast_memStr+' && kmercountexact.sh mincount=%d overwrite=true fastadump=f in=%s out=%s k=%d'%(min_count,subgenome_labels[subgenome_name],work_dir+subgenome_name+'.kcount',kmer_length),shell=True)
            else:
                subprocess.call('jellyfish count -L %d -m %d -s %d -t 15 -C -o %s/mer_counts.jf %s && jellyfish dump %s/mer_counts.jf -c > %s/%s'%(min_count,kmer_length,os.stat(subgenome_labels[subgenome_name]).st_size,work_dir,subgenome_labels[subgenome_name],work_dir,work_dir,subgenome_name+'.kcount'))
        kmer_matrix.append(return_kcounts(subgenome_name))#dict(zip(kmer_counts[0],kmer_counts[1])))
    print subgenome_labels.keys()
    #dv = DictVectorizer(sparse=True)
    click.echo('fitting')
    #dv.fit(kmer_matrix)
    kmers = list(set().union(*map(dict.keys,kmer_matrix)))#reduce(set.union, map(set, map(dict.keys,kmer_matrix)))
    print kmers[0:10]
    kmer_mapping = LabelEncoder()
    kmer_mapping.fit(kmers)
    subgenomes = np.array(subgenome_labels.keys())
    data = sps.dok_matrix((len(subgenomes),len(kmers)),dtype=np.int)
    for i,d in enumerate(kmer_matrix):
        print i
        data[i,kmer_mapping.transform(d.keys())] = np.array(d.values())
    del kmer_matrix
    # FIXME I can use above to speed up other algorithms, kmer matrix generation
    data=data.tocsr()
    pickle.dump(subgenomes,open(work_dir+'subgenomes_all.p','wb'))
    if fast_tree: #FIXME visualize results?
        pca_data = Pipeline([('scaler',StandardScaler(with_mean=False)),('kpca',KernelPCA(n_components=3))]).fit_transform(data)
        np.save(work_dir+'kmer_pca_results.npy',pca_data)
        d_matrix = pairwise_distances(pca_data)
        d_matrix = (d_matrix + d_matrix.T)/2
        pd.DataFrame(d_matrix,index=subgenome_labels.keys(),columns=subgenome_labels.keys()).to_csv(work_dir+'kmer_distance_matrix.csv')
        matrix = [row[:i+1] for i,row in enumerate(d_matrix.tolist())]#[list(distance_matrix[i,0:i+1]) for i in range(distance_matrix.shape[0])]
        dm = _DistanceMatrix(names=subgenome_labels.keys(),matrix=matrix)
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(dm)
        Phylo.write(tree,work_dir+'kmer_guide_tree.nh','newick')
        ctx.invoke(plotPositions,positions_npy=work_dir+'kmer_pca_results.npy',labels_pickle=work_dir+'subgenomes_all.p',colors_pickle='',output_fname=work_dir+'subgenome_bio_pca.html')
    data = data.T
    #kmers = np.array(dv.feature_names_)
    #kmer_matrix = dv.transform(kmer_matrix).T
    #row_function = lambda l: np.max(l)/float(np.min(l)) >= ratio_threshold
    #threshold_bool_array = []
    #for i in range(data.shape[0]): #FIXME here is where the slowdown occurs
    #    threshold_bool_array.append(row_function(np.clip(data[i,:].toarray()[0],default_kcount_value,None)))
    print 'pretransform'
    #print data.max(axis=1).toarray()[:,0][0:10]
    #print np.clip(data.min(axis=1).toarray(),default_kcount_value,None)[:,0][0:10]
    max_array = data.max(axis=1).toarray()[:,0]
    print sum(max_array >= max_val)
    threshold_bool_array = np.logical_and(np.vectorize(lambda x: x >= ratio_threshold)(max_array/np.clip(data.min(axis=1).toarray()[:,0],default_kcount_value,None)), max_array >= max_val)
    print 'post threshold'
    click.echo('check')
    #threshold_bool_array = np.array(threshold_bool_array)
    print threshold_bool_array
    kmers = kmer_mapping.inverse_transform(np.arange(data.shape[0]))[threshold_bool_array]#np.array(kmers)[kmer_mapping.transform(kmers)[threshold_bool_array]] # FIXME
    data = data[threshold_bool_array]
    pickle.dump(kmers,open(work_dir+'kmers_union.p','wb'))
    sps.save_npz(work_dir+'union_kmer_matrix.npz',data)
    df = pd.SparseDataFrame(data,default_fill_value=0,index=kmers,columns=subgenomes).to_dense()
    df.to_csv(work_dir+'kmer_master_count_matrix.csv')
    #np.apply_along_axis(row_function,1,[np.clip(kmer_matrix[i,:].toarray()[0],default_value,None) for i in range(kmer_matrix.shape[0])])
    #for i in range(kmer_matrix.shape[0]):
    #    row = np.clip(kmer_matrix[i,:].toarray(),default_value,None)
    # FIXME can now filter array and store as pandas to excel


@utilities.command() # fixme future for polyCRACKER/metacracker, can use ubiquitous kmers between many lines, on non ubiquitous, rated via tfidf to definte kmers used for polycracker pipeline, automatic dimensionality reduction via abundance in different lines
@click.pass_context
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for plotting kmer count matrix is done.', type=click.Path(exists=False))
@click.option('-csv', '--kmer_count_matrix', default='./kmer_master_count_matrix.csv', show_default=True, help='Kmer count matrix.', type=click.Path(exists=False))
@click.option('-r', '--reduction_method', default='tsne', show_default=True, help='Type of dimensionality reduction technique to use.', type=click.Choice(['tsne','kpca', 'spectral', 'nmf', 'feature']))
@click.option('-d', '--n_dimensions', default=3, help='Number of dimensions to reduce to.', show_default=True)
@click.option('-k', '--kernel', default='cosine', help='Kernel for KPCA. Cosine is particularly effective.', type=click.Choice(['linear','poly','rbf','sigmoid','cosine']), show_default=True)
@click.option('-m', '--metric', default='cosine', help='Distance metric used to compute distance matrix.', type=click.Choice(['cityblock','cosine','euclidean','l1','l2','manhattan','braycurtis','canberra','chebyshev','dice','hamming','haversine','infinity','jaccard','kulsinski','mahalanobis','matching','minkowski','rogerstanimoto','russellrao','seuclidean','sokalmichener','sokalsneath','wminkowski','ectd']), show_default=True)
@click.option('-min_c', '--min_cluster_size', default = 300, show_default=True, help='Minimum cluster size for hdbscan algorithm.')
@click.option('-min_s', '--min_samples', default = 150, show_default=True, help='Minimum number samples for hdbscan algorithm.')
@click.option('-nn', '--n_neighbors', default=10, help='Number of nearest neighbors in generation of nearest neighbor graph.', show_default=True)
@click.option('-scan','--hyperparameter_scan', is_flag=True, help='Whether to conduct hyperparameter scan of best clustering parameters.')
@click.option('-min_n', '--min_number_clusters', default = 3, show_default=True, help='Minimum number of clusters to find if doing hyperparameter scan. Parameters that produce fewer clusters face heavy penalization.')
@click.option('-low', '--low_counts', default='5,5,5', show_default=True, help='Comma delimited list of low bound on min_cluster_size, min_samples, and nearest neighbors respectively for hyperparameter scan.')
@click.option('-j', '--n_jobs', default=1, help='Number of jobs for TSNE transform.', show_default=True)
@click.option('-s', '--silhouette', is_flag=True, help='Use mean silhouette coefficient with mahalanobis distance as scoring metric for GA hyperparameter scan.')
@click.option('-v', '--validity', is_flag=True, help='Use hdbscan validity metric for clustering score. Takes precedent over silhouette score.')
def explore_kmers(ctx,work_dir,kmer_count_matrix,reduction_method,n_dimensions, kernel, metric, min_cluster_size,min_samples, n_neighbors, hyperparameter_scan, min_number_clusters, low_counts, n_jobs, silhouette, validity):
    """Perform dimensionality reduction on matrix of kmer counts versus genomes and cluster using a combination of genetic algorithm + density based clustering (hdbscan). Uncovers kmer conservation patterns/rules across different lines."""
    import hdbscan
    from sklearn.manifold import SpectralEmbedding#, TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.metrics import calinski_harabaz_score, silhouette_score
    from evolutionary_search import maximize
    import seaborn as sns
    # FIXME mahalanobis arguments to pass V, debug distance metrics
    # FIXME add ECTD as a distance metric? Feed as precomputed into hdbscan?
    hdbscan_metric = (metric if metric not in ['ectd','cosine','mahalanobis'] else 'precomputed')
    optional_arguments = dict()

    def cluster_data(t_data,min_cluster_size, min_samples, cluster_selection_method= 'eom'): # , n_neighbors = n_neighbors
        #print min_cluster_size, min_samples, kernel, n_neighbors
        labels = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples= min_samples, cluster_selection_method=cluster_selection_method, metric = hdbscan_metric, alpha = 1.0).fit_predict(t_data)
        #lp = LabelPropagation(kernel=kernel, n_neighbors = n_neighbors) # FIXME Try decision trees next, maybe just use highest chi square valued ones for training
        #lp.fit(t_data,labels) #kmer_count_matrix
        #labels = np.array(lp.predict(t_data))
        return labels
    scoring_method = lambda X, y: hdbscan.validity.validity_index(X,y,metric=hdbscan_metric) if validity else (silhouette_score(X,y,metric='precomputed' if metric =='mahalanobis' else 'mahalanobis',V=(np.cov(X,rowvar=False) if metric != 'mahalanobis' else '')) if silhouette else calinski_harabaz_score(X,y))

    def return_cluster_score(t_data,min_cluster_size, min_samples, cluster_selection_method): # , n_neighbors
        click.echo(' '.join(map(str,[min_cluster_size, min_samples, cluster_selection_method]))) # , n_neighbors
        labels = cluster_data(t_data,min_cluster_size, min_samples, cluster_selection_method) # , n_neighbors
        n_clusters = labels.max() + 1
        X = t_data if validity else t_data[labels != -1,:]
        y = labels if validity else labels[labels != -1]
        return scoring_method(X,y)/((1. if n_clusters >= min_number_clusters else float(min_number_clusters - n_clusters + 1))*(1.+len(labels[labels == -1])/float(len(labels)))) #FIXME maybe change t_data[labels != -1,:], labels[labels != -1] and (1.+len(labels[labels == -1])/float(len(labels)))

    def ectd_graph(t_data):
        paired = pairwise_distances(t_data)
        neigh = NearestNeighbors(n_neighbors=n_neighbors,metric='precomputed')
        neigh.fit(t_data)
        fit_data = neigh.kneighbors_graph(paired, mode='distance')
        fit_data = ( fit_data + fit_data.T )/2.
        min_span_tree = sps.csgraph.minimum_spanning_tree(paired).to_csc()
        min_span_tree = ( min_span_tree + min_span_tree.T )/2.
        return fit_data + min_span_tree

    sns.set(style="ticks")
    low_counts = map(int,low_counts.split(','))
    kmer_count_matrix = pd.read_csv(kmer_count_matrix,index_col = ['Unnamed: 0'])
    subgenome_names = list(kmer_count_matrix)
    labels_text = []
    for index, row in kmer_count_matrix.iterrows():
        labels_text.append(index+'<br>'+'<br>'.join(['%s: %s'%(subgenome,val) for subgenome,val in zip(subgenome_names,row.as_matrix().astype(str).tolist())]))#+'-'+','.join(subgenome_names+row.as_matrix().astype(str).tolist())) fixme ' ' ', '
    pickle.dump(np.array(labels_text),open(work_dir+'kmer_labels_coordinates.p','wb'))
    pickle.dump(kmer_count_matrix.idxmax(axis=1).as_matrix(),open(work_dir+'kmer_labels.p','wb'))
    transform_dict = {'kpca':KernelPCA(n_components=n_dimensions, kernel=kernel),'spectral':SpectralEmbedding(n_components=n_dimensions),'nmf':NMF(n_components=n_dimensions),'tsne':TSNE(n_components=n_dimensions,n_jobs=n_jobs), 'feature': FeatureAgglomeration(n_clusters=n_dimensions)}
    t_data = Pipeline([('scaler',StandardScaler(with_mean=False)),(reduction_method,transform_dict[reduction_method])]).fit_transform(kmer_count_matrix) # ,('spectral_induced',SpectralEmbedding(n_components=3))
    np.save(work_dir+'kmer_counts_pca_results.npy',t_data)
    if metric == 'ectd':
        t_data = ectd_graph(t_data)
    elif metric == 'cosine':
        t_data = pairwise_distances(t_data,metric='cosine')
    elif metric == 'mahalanobis':
        t_data = pairwise_distances(t_data,metric='mahalanobis',V=np.cov(t_data,rowvar=True))
    ctx.invoke(plotPositions,positions_npy=work_dir+'kmer_counts_pca_results.npy',labels_pickle=work_dir+'kmer_labels_coordinates.p',colors_pickle=work_dir+'kmer_labels.p',output_fname=work_dir+'subgenome_bio_rules_kmers.html')
    if hyperparameter_scan: # FIXME Test hyperparameter_scan, add ability to use label propagation, or scan for it or iterations of label propagation
        best_params, best_score, score_results, hist, log = maximize(return_cluster_score, dict(min_cluster_size = np.unique(np.linspace(low_counts[0],min_cluster_size,10).astype(int)).tolist(), min_samples = np.unique(np.linspace(low_counts[1],min_samples, 10).astype(int)).tolist(), cluster_selection_method= ['eom', 'leaf'] ), dict(t_data=t_data), verbose=True, n_jobs = 1, generations_number=8, gene_mutation_prob=0.45, gene_crossover_prob = 0.45) # n_neighbors = np.unique(np.linspace(low_counts[2], n_neighbors, 10).astype(int)).tolist()),
        labels = cluster_data(t_data,min_cluster_size=best_params['min_cluster_size'], min_samples=best_params['min_samples'], cluster_selection_method= best_params['cluster_selection_method']) # , n_neighbors = best_params['n_neighbors']
        print best_params, best_score, score_results, hist, log
    else:
        labels = cluster_data(t_data,min_cluster_size, min_samples)
    rules_orig = np.vectorize(lambda x: 'rule %d'%x)(labels)
    pickle.dump(rules_orig,open(work_dir+'rules.p','wb'))
    ctx.invoke(plotPositions,positions_npy=work_dir+'kmer_counts_pca_results.npy',labels_pickle=work_dir+'kmer_labels_coordinates.p',colors_pickle=work_dir+'rules.p',output_fname=work_dir+'subgenome_bio_rules_propagated_kmers.html')
    observed = kmer_count_matrix.as_matrix()
    print observed
    rules = np.array([rules_orig[idx] for idx in sorted(np.unique(rules_orig,return_index=True)[1])])
    rules_mean = [observed[rules_orig==rule,:]/observed[rules_orig==rule,:].sum(axis=1).astype(float)[:,None] for rule in rules]
    print rules_mean
    """
    for i, rule in enumerate(rules_mean):
        rule_name = rules[i].replace(' ','')
        rule = pd.DataFrame(rule,columns=subgenome_names).melt(id_vars = None, var_name='Genome Name', value_name='Normalized Counts')
        plt.figure()
        sns.boxplot(x='Normalized Counts',y='Genome Name', data=rule, palette="vlag") #whis=np.inf,
        sns.swarmplot(x='Normalized Counts', y='Genome Name', data=rule,
              size=2, color=".3", linewidth=0)
        sns.despine(trim=True, left=True)
        plt.savefig(work_dir+rule_name+'_genome_dominance.png',dpi=300)
    """ # FIXME use this to combine rules if they have matching normalized distributions, pairwise chi-square clustering of rules
    pd.DataFrame([rule.mean(axis=0) for rule in rules_mean],index=rules,columns=subgenome_names).to_csv(work_dir+'kmers_unsupervised_rules.csv')

    n_subgenomes = len(subgenome_names)
    #sum_count = kmer_count_matrix.as_matrix().sum(axis=1)
    print n_subgenomes
    expected = np.ones(observed.shape) * np.sum(observed,axis=0)/np.sum(observed,axis=0).sum() * observed.sum(axis=1)[:,np.newaxis] #np.ones(kmer_count_matrix.shape)/float(n_subgenomes) * kmer_count_matrix.as_matrix().sum(axis=1)[:,None]
    print expected
    chi2_arr = (observed - expected)**2/expected#np.apply_along_axis(lambda x: (x-expected*np.sum(x))**2/(expected*np.sum(x)),1,kmer_count_matrix.as_matrix())
    chi2_sum = chi2_arr.sum(axis=1)
    df2 = pd.DataFrame(chi2_arr,index=kmer_count_matrix.index.values,columns=[name+'-chi2' for name in subgenome_names])
    df3 = pd.concat([kmer_count_matrix,df2],axis=1,join='outer')
    df2['Sum'] = chi2_sum
    print df2, df3
    kmer_count_matrix['Rules'] = rules_orig
    #kmer_count_matrix = kmer_count_matrix.set_index(['Unnamed: 0'])
    kmer_count_matrix.to_csv(work_dir + 'kmer_master_matrix_rules.csv')
    for folder in ['rules_plots/','rules_plots_chi2/']:
        try:
            os.makedirs(work_dir+folder)
        except:
            pass
    ctx.invoke(plot_rules,work_dir=work_dir+'rules_plots/',rule_csv=work_dir + 'kmer_master_matrix_rules.csv')
    df2.to_csv(work_dir+'kmer_master_chi2_matrix.csv')
    df3.to_csv(work_dir+'kmer_master_count_chi2_matrix.csv')
    t_data = Pipeline([('scaler',StandardScaler(with_mean=False)),(reduction_method,transform_dict[reduction_method])]).fit_transform(df3)
    print t_data
    np.save(work_dir+'kmer_counts_pca_chi2_results.npy',t_data)
    if metric == 'ectd':
        t_data = ectd_graph(t_data)
    elif metric == 'cosine':
        t_data = pairwise_distances(t_data,metric='cosine')
    elif metric == 'mahalanobis':
        t_data = pairwise_distances(t_data,metric='mahalanobis',V=np.cov(t_data,rowvar=True))
    if hyperparameter_scan: # FIXME Test hyperparameter_scan, add ability to use label propagation, or scan for it or iterations of label propagation
        best_params, best_score, score_results, hist, log = maximize(return_cluster_score, dict(min_cluster_size = np.unique(np.linspace(low_counts[0],min_cluster_size,10).astype(int)).tolist(), min_samples = np.unique(np.linspace(low_counts[1],min_samples, 10).astype(int)).tolist(), cluster_selection_method= ['eom', 'leaf'] ), dict(t_data=t_data), verbose=True, n_jobs = 1, generations_number=8, gene_mutation_prob=0.45, gene_crossover_prob = 0.45) # n_neighbors = np.unique(np.linspace(low_counts[2], n_neighbors, 10).astype(int)).tolist()),
        labels = cluster_data(t_data,min_cluster_size=best_params['min_cluster_size'], min_samples=best_params['min_samples'], cluster_selection_method= best_params['cluster_selection_method']) # , n_neighbors = best_params['n_neighbors']
        print best_params, best_score, score_results, hist, log
    else:
        labels = cluster_data(t_data,min_cluster_size, min_samples)
    rules = np.vectorize(lambda x: 'rule %d'%x)(labels)
    df3['Rules'] = rules
    pickle.dump(rules,open(work_dir+'rules_chi2.p','wb'))
    ctx.invoke(plotPositions,positions_npy=work_dir+'kmer_counts_pca_chi2_results.npy',labels_pickle=work_dir+'kmer_labels_coordinates.p',colors_pickle=work_dir+'rules_chi2.p',output_fname=work_dir+'subgenome_bio_rules_kmers_chi2.html')
    df3.to_csv(work_dir+'kmer_master_count_chi2_matrix_rules.csv')
    ctx.invoke(plot_rules,work_dir=work_dir+'rules_plots_chi2/',rule_csv=work_dir + 'kmer_master_count_chi2_matrix_rules.csv')
    X = [observed[labels==label,:]/observed[labels==label,:] for label in range(labels.max()+1)]
    X_mean = np.vstack([x_s.mean(axis=0) for x_s in X])
    X_std = np.vstack([x_s.std(axis=0) for x_s in X])
    pd.DataFrame(X_mean,index=['rule_%d'%label for label in range(labels.max()+1)],columns=subgenome_names).to_csv(work_dir+'kmers_unsupervised_rules_chi2.csv')
    pd.DataFrame(X_std,index=['rule_%d'%label for label in range(labels.max()+1)],columns=subgenome_names).to_csv(work_dir+'kmers_unsupervised_rules_chi2_uncertainty.csv')
    # FIXME function: turn above into merging and splitting clusters based on heterogeneity (maybe use local PCA?), though no order, but maybe find way to order based on x-y-z
    # FIXME function: add ability to input consensus sequences and start to find rule and subclass breakdown, maybe subclass breakdown for each rule
    # FIXME fix above merging clusters method
    # FIXME add function to use user input for rule discovery, hypotheses user is looking for can also help with dimensionality breakdown and initial clustering, as well as postprocessing
    # FIXME comment and clean, also maybe make more command groups
    # FIXME check out penalization for genetic algorithm
    # FIXME add rule finding http://skope-rules.readthedocs.io/en/latest/auto_examples/plot_skope_rules.html#sphx-glr-auto-examples-plot-skope-rules-py , way to display rules http://skope-rules.readthedocs.io/en/latest/skope_rules.html
    # FIXME Euclidean commute time as distance metric, -> clustering via random walks
    # FIXME SVD + DQC instead of TSNE + HDBSCAN ?


@utilities.command()
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for plotting kmer count matrix is done.', type=click.Path(exists=False))
def find_rules(work_dir):
    """In development: Elucidate underlying rules of certain kmer conservation patterns."""
    import Orange
    from sklearn import tree
    df = pd.read_csv(work_dir+'kmer_master_count_chi2_matrix_rules.csv')
    X , y = df.iloc[:,1:-1].as_matrix(), df.iloc[:,-1].as_matrix()
    n_col = X.shape[1]
    print n_col
    X1 = X[:,:n_col/2]
    #print X1
    X1 /= X1.sum(axis=1).astype(float)[:, np.newaxis]
    X[:,:n_col/2] = X1
    #t_data = FeatureAgglomeration(n_clusters=n_col/2)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X[df['Rules'].as_matrix() != 'rule -1'], y[df['Rules'].as_matrix() != 'rule -1'])
    tree.export_graphviz(clf, out_file=work_dir+'rules_graph.dot',
                         feature_names=list(df)[1:-1],
                         class_names=np.unique(y[df['Rules'].as_matrix() != 'rule -1']),
                         filled=True, rounded=True,
                         special_characters=True)
    subprocess.call('dot -Tpng %s -o %s'%(work_dir+'rules_graph.dot',work_dir+'rules_graph.png'),shell=True)
    df = df.rename(columns=dict(zip(list(df),['i#k-mer']+['C#'+ val for val in df.columns.values[1:-1]]+['cD#'+df.columns.values[-1]])))
    #print X1, df.iloc[:,1:n_col/2+1]
    df.iloc[:,1:n_col/2+1] = X1
    print df
    df.to_csv(work_dir+'orange_input_rules.csv',index=False, sep='\t')
    subprocess.call("grep -v 'rule -1' %s > %s"%(work_dir+'orange_input_rules.csv',work_dir+'orange_input_rules_no_out.csv'),shell=True)
    data = Orange.data.Table(work_dir+'orange_input_rules_no_out.csv')
    print data
    cn2_classifier = Orange.classification.rules.CN2Learner(data)
    with open(work_dir+'rules.txt','w') as f:
        for r in cn2_classifier.rules:
            f.write(Orange.classification.rules.rule_to_string(r)+'\n')


@utilities.command()
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for plotting kmer count matrix is done.', type=click.Path(exists=False))
@click.option('-n', '-n_rules', default = 10, show_default=True, help='Number of rules to find.')
def find_rules2(work_dir, n_rules):
    """In development: Elucidate underlying rules of certain kmer conservation patterns."""
    from skrules import SkopeRules
    rng = np.random.RandomState(42)
    df = pd.read_csv(work_dir+'kmer_master_count_chi2_matrix_rules.csv')
    clf = SkopeRules(feature_names = df.columns.values[:-1],random_state=rng, n_estimators=10, max_samples = 1)
    X , y = df.iloc[:,:-1].as_matrix(), df.iloc[:,-1].as_matrix()
    clf.fit(X[df['Rules'] != 'rule -1'],y[df['Rules'] != 'rule -1'])
    labels=clf.predict(X)
    for rule in clf.rules_[:n_rules]:
        print(rule[0])

@utilities.command()
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for plotting kmer count matrix is done.', type=click.Path(exists=False))
@click.option('-csv', '--rule_csv', default = './kmer_master_count_chi2_matrix_rules.csv', show_default = True, help='Kmer count matrix with appended rule labels. Default includes chi-squared values.', type=click.Path(exists=False))
def plot_rules(work_dir,rule_csv):
    """Plots normalized frequency distribution of select kmers across multiple genomes of certain rules/patterns."""
    df = pd.read_csv(rule_csv)
    X , y = df.iloc[:,1:-1].as_matrix(), df.iloc[:,-1].as_matrix()
    if 'kmer_master_count_chi2_matrix_rules.csv' in rule_csv:
        n_col = X.shape[1]
        X1 = X[:,:n_col/2]
        X1 /= X1.sum(axis=1).astype(float)[:, np.newaxis]
        #X[:,:n_col/2] = X1
    else:
        X1 = X / X.sum(axis=1)[:, np.newaxis].astype(float)
    col_names = list(df)[1:-1]
    for rule in set(y) - {'rule -1'}:
        plots = []
        x = X1[y == rule,:]
        for i, col in enumerate(col_names):
            plots.append(go.Box(y=x[:,i],name=col,boxpoints='all'))
        py.plot(go.Figure(data=plots),filename=work_dir+rule.replace(' ','_')+'.html',auto_open=False)


@utilities.command()
def build_pairwise_align_similarity_structure(fasta_in):
    """Take consensus repeats, generate graph structure from sequence similarity, and cluster. Can be useful to find homologous repeats from different genomes."""
    from pyfaidx import Fasta
    from Bio import pairwise2
    f = Fasta(fasta_in)
    repeat_names = f.keys()
    similarity_mat = np.ones((len(repeat_names),)*2,dtype=np.float)
    diagonal_elements = zip(range(len(repeat_names)),range(len(repeat_names)))
    for i,j in list(combinations(range(len(repeat_names)),2)) + diagonal_elements:
        similarity_mat[i,j] = pairwise2.align.globalxx(str(f[repeat_names[i]][:]),str(f[repeat_names[j]][:]),score_only=True)
        if i != j:
            similarity_mat[j,i] = similarity_mat[i,j]
    for i,j in list(combinations(repeat_names,2)):# + diagonal_elements:
        similarity_mat[i,j] = similarity_mat[i,j]/np.sqrt(similarity_mat[i,i]*similarity_mat[j,j])
        if i != j:
            similarity_mat[j,i] = similarity_mat[i,j]
    #dissimilarity_matrix = 1. - similarity_mat
    #nn = NearestNeighbors(n_neighbors= 10,metric='precomputed').fit(dissimilarity_matrix)
    spec = SpectralClustering(n_clusters = 8, affinity='precomputed')
    labels = spec.fit_predict(similarity_mat)
    return repeat_names , similarity_mat, labels


@utilities.command()
@click.option('-i', '--input_files', default = "genomes/*.fa", show_default=True, help="Input fasta files. Put input in quotations if specifying multiple files. Eg. 'genomes/*.fasta'", type=click.Path(exists=False))
@click.option('-kl', '--kmer_length', default = 23, show_default=True, help='Default kmer length for signature.')
@click.option('-o', '--output_file_prefix', default = 'output', show_default=True, help='Output file prefix, default outputs output.cmp.dnd and output.sig.', type=click.Path(exists=False) )
@click.option('-mf', '--multi_fasta', is_flag=True, help='Instead of using multiple fasta files, generate signatures based on sequences in a multifasta file.')
@click.option('-s','--scaled', default = 10000, show_default=True, help='Scale factor for sourmash.')
def generate_genome_signatures(input_files, kmer_length, output_file_prefix,multi_fasta,scaled):
    """Wrapper for sourmash. Generate genome signatures and corresponding distance matrices for scaffolds/fasta files."""
    subprocess.call('sourmash compute -f --scaled %d %s -o %s.sig -k %d %s'%(scaled,input_files,output_file_prefix,kmer_length,'--singleton' if multi_fasta else ''),shell=True)
    subprocess.call('sourmash compare %s.sig --csv %s.cmp.csv'%(output_file_prefix,output_file_prefix),shell=True)


@utilities.command(name='color_trees')
@click.option('-w', '--work_dir', default='./TE_subclass_analysis/', show_default=True, help='Work directory for computations of Trees.', type=click.Path(exists=False))
@click.option('-n', '--newick_in', default='./x.treefile', help='Input tree, newick format.', show_default=True, type=click.Path(exists=False))
@click.option('-rd', '--top_repeat_dict_pickle', default='./TE_cluster_analysis/informative_subgenome_TE_dict.p', help='Pickle containing dict of top informative subgenome differential TEs by subgenome.', show_default=True, type=click.Path(exists=False))
def color_trees(work_dir,newick_in,top_repeat_dict_pickle):
    """Color phylogenetic trees by progenitor of origin."""
    from ete3 import Tree, TreeStyle, TextFace
    #from xvfbwrapper import Xvfb
    work_dir+='/'
    ts = TreeStyle()
    tree =  Tree(newick_in)
    colors  = ["red","green","blue","orange","purple","pink","yellow"]
    subgenomes_repeat_dict = pickle.load(open(top_repeat_dict_pickle,'r'))
    subgenomes = subgenomes_repeat_dict.keys()
    subgenome_colors = dict(zip(subgenomes,colors[:len(subgenomes)]))
    repeat_color_dict = { repeat:subgenome_colors[s] for s,repeat in reduce(lambda x,y: x+y,[map(lambda x: (subgenome,x),subgenomes_repeat_dict[subgenome]) for subgenome in subgenomes_repeat_dict])}
    for leaf in tree:
        leaf.img_style['size'] = 0
        if leaf.is_leaf():
            name = leaf.name
            if 'Simple_repeat' in name and 'n' in name:
                name = '(' + name.split('_')[1] + ')n#'+'Simple_repeat'
            else:
                name = name.rsplit('-',1)[0]+'-'+'#'.join(name.rsplit('-',1)[1].split('_',1)).replace('_','/')
            print name
            color=repeat_color_dict.get(name, None)
            #print color
            if color:
                name_face = TextFace(name,fgcolor=color)
                leaf.add_face(name_face,column=0,position='branch-right')
    ts.show_leaf_name = False
    ts.show_branch_support = True
    R = tree.get_midpoint_outgroup()
    tree.set_outgroup(R)
    tree.render(work_dir+"subclass_%s_tree.png"%(newick_in[newick_in.rfind('/')+1:newick_in.rfind('.')]), dpi=300, w=1000, tree_style=ts)
    tree.render(work_dir+"subclass_%s_tree.pdf"%(newick_in[newick_in.rfind('/')+1:newick_in.rfind('.')]), dpi=300, w=1000, tree_style=ts)


@utilities.command(name='dash_genome_quality_assessment')
@click.pass_context
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for finding array of kmers versus prescaffolded genomes with unassigned sequences.', type=click.Path(exists=False))
@click.option('-csv', '--kmer_rule_matrix', default = './kmer_master_matrix_rules.csv', show_default = True, help='Kmer count matrix with appended rule labels.', type=click.Path(exists=False))
@click.option('-g','--pre_scaffolded_genomes', default = 'genome1:./path/genome_1_fname.fa,genome2:./path/genome_2_fname.fa,genome3:./path/genome_3_fname.fa',show_default=True,help="Comma delimited dictionary of genome names and corresponding genomes. Genomes should be assemblies prior to polyCRACKER/bbtools binning and scaffolding. Can input './path/*.fa[sta]' to input all genomes in path.",type=click.Path(exists=False))
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
def dash_genome_quality_assessment(ctx,work_dir,kmer_rule_matrix,pre_scaffolded_genomes,blast_mem):
    """Input pre chromosome level scaffolded genomes, before split by ploidy, and outputs frequency distribution of select kmers.
    Used to compare this distribution of kmers to that found from comparing post-scaffolded polyploid subgenomes.
    The scaffolding, if reference based, may be biased and missing key repeat information, so comparing to prescaffolded genomes may show whether repeat info was lost during scaffolding for important words."""
    import glob
    work_dir += '/'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    df = pd.read_csv(kmer_rule_matrix, index_col = 0)
    genomes = dict(map(lambda x: tuple(x.split(':')),pre_scaffolded_genomes.split(','))) if '*' not in pre_scaffolded_genomes else {genome.rsplit('/')[-1]:genome for genome in glob.glob(pre_scaffolded_genomes)}
    print genomes
    kmers = list(df.index)
    with open(work_dir+'kmer_fasta.fa','w') as f:
        f.write('\n'.join(['>%s\n%s'%(kmer,kmer) for kmer in kmers]))
    genomes_dict = {k:{kmer:0 for kmer in kmers} for k in genomes.keys()}
    df2 = pd.DataFrame(df['Rules'],index=kmers,columns=['Rules'])
    #for genome in genomes_dict:
    #    df2[genome] = 0
    kmer_length = int(np.mean(map(len,kmers)))
    blast_memStr = "export _JAVA_OPTIONS='-Xms5G -Xmx%sG'"%(blast_mem)
    for genome in genomes:
        subprocess.call(blast_memStr+' && kmercountexact.sh mincount=%d overwrite=true fastadump=f in=%s out=%s k=%d'%(5,genomes[genome],work_dir+genome+'.kcount',kmer_length),shell=True)
        # fixme kmers found from bbmap are different than ones found in kmercountexact
        #ctx.invoke(blast_kmers,blast_mem=blast_mem,reference_genome=genomes[genome],kmer_fasta=work_dir+'kmer_fasta.fa', output_file=work_dir+genome+'_blasted.sam',kmer_length=8)
        #genomes_dict[genome].update(Counter(os.popen("awk '{ print $1 }' %s"%(work_dir+genome+'_blasted.sam')).read().splitlines()))
        genomes_dict[genome].update(dict(zip(os.popen("awk '{ print $1 }' %s"%work_dir+genome+'.kcount').read().splitlines(),os.popen("awk '{ print $2 }' %s"%work_dir+genome+'.kcount').read().splitlines())))
        df2[genome] = pd.DataFrame.from_dict(genomes_dict[genome],orient='index',dtype=np.int).reindex(index=kmers)
    df2.to_csv(work_dir+'genome_quality_check.csv')


@utilities.command()
@click.pass_context
@click.option('-p','--scaffolds_pickle', default='scaffolds_stats.p', show_default=True, help='Pickle file generated from final_stats. Contains all scaffolds.',  type=click.Path(exists=False))
@click.option('-f', '--fasta_results', default='./cluster-0.fa,./cluster-1.fa',show_default=True, help= "Fasta files of clusters output from scimm/metabat. Write './*.fa' or 'folder/*.fasta' if specifying multiple files.",  type=click.Path(exists=False))
@click.option('-p1','--scimm_metabat_pickle', default='scaffolds_stats.scimm.metabat.labels.p', show_default=True, help='Pickle file generated from final_stats. These are polycracker results.',  type=click.Path(exists=False))
@click.option('-p2','--progenitors_pickle', default='scaffolds_stats.progenitors.labels.p', show_default=True, help='Pickle file generated from final_stats. These are progenitor mapping results.',  type=click.Path(exists=False))
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for scimm/metabat comparison is done. Make sure only text files are subgenome outputs in folder.', type=click.Path(exists=False))
def compare_scimm_metabat(ctx,scaffolds_pickle,fasta_results,scimm_metabat_pickle, progenitors_pickle, work_dir):
    from sklearn.metrics import cohen_kappa_score, homogeneity_completeness_v_measure
    from sklearn.preprocessing import LabelEncoder
    import glob
    work_dir += '/'
    #print fasta_results
    def cohen_max(y_true,y_pred,weights):
        y_range = range(y_pred.max()+1)
        y_pred_permutations = permutations(y_range)
        cohens = []
        for perm in y_pred_permutations:
            d = dict(zip(y_range,perm))
            y_p = np.vectorize(lambda x: d[x])(y_pred)
            cohen = cohen_kappa_score(y_true,y_p, sample_weight = weights)
            print cohen
            cohens.append(cohen)
        return max(cohens)
    scaffold_len = lambda scaffold: int(scaffold.split('_')[-1])-int(scaffold.split('_')[-2])
    try:
        os.makedirs(work_dir)
    except:
        pass
    try:
        os.makedirs(work_dir+'results_compare/')
    except:
        pass
    if fasta_results.endswith('*.fa') or fasta_results.endswith('*.fasta'):
        fasta_results = glob.glob(fasta_results)
    else:
        fasta_results=fasta_results.split(',')
    for i,fasta in enumerate(fasta_results): #fixme throw in some metabat
        f1 = Fasta(fasta)
        with open(work_dir+'subgenome_%d.txt'%i,'w') as f:
            f.write('\n'.join(f1.keys()))
    ctx.invoke(convert_subgenome_output_to_pickle,input_dir=work_dir,scaffolds_pickle=scaffolds_pickle,output_pickle=scimm_metabat_pickle)
    weights = np.vectorize(scaffold_len)(pickle.load(open(scaffolds_pickle,'rb')))
    scimm_metabat_labels = pickle.load(open(scimm_metabat_pickle,'rb'))
    progenitor_labels = pickle.load(open(progenitors_pickle,'rb'))
    progenitor_labels = LabelEncoder().fit_transform(progenitor_labels)
    scimm_metabat_labels = LabelEncoder().fit_transform(scimm_metabat_labels)
    w = weights.tolist()
    weights_mode = max(w, key = w.count)
    output_dict = dict(cohen_kappa_score=cohen_max(progenitor_labels,scimm_metabat_labels,weights),homogeneity_completeness_v_measure=homogeneity_completeness_v_measure(progenitor_labels,scimm_metabat_labels),weights_len = weights[weights==weights_mode].sum()/float(sum(weights))) #fixme permute labels here!, add weights, etc))
    with open(work_dir+'results_compare/results.txt','w') as f:
        f.write(str(output_dict))


@utilities.command(name='return_dash_data_structures')
@click.pass_context
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
@click.option('-r', '--reference_genome', help='Genome to blast against.', type=click.Path(exists=False))
@click.option('-o', '--output_dir', default='./', show_default=True, help='Work directory for computations and outputs.', type=click.Path(exists=False))
@click.option('-sl', '--split_length', default=75000, help='Length of intervals in bedgraph files.', show_default=True)
@click.option('-npy', '--kmer_pca_data', default='./kmer_counts_pca_results.npy', show_default=True, help='Location of pca of kmer count matrix.', type=click.Path(exists=False))
@click.option('-kl', '--kmer_pca_labels', default='./kmer_labels_coordinates.p', show_default=True, help='Location of kmer labels pickle for each pca datapoint, eg. ATTTCGGCGAT Bd: 10, Bs: 20, ABRD: 40.', type=click.Path(exists=False))
@click.option('-kr', '--kmer_rules_pickle', default='./rules.p', show_default=True, help='Location of kmer rules pickle for each pca datapoint, eg.rule 3.', type=click.Path(exists=False))
@click.option('-csv', '--kmer_rule_matrix', default = './kmer_master_count_chi2_matrix_rules.csv', show_default = True, help='Kmer count matrix with appended rule labels. Default includes chi-squared values.', type=click.Path(exists=False))
def return_dash_data_structures(ctx,blast_mem,reference_genome,output_dir,split_length,kmer_pca_data,kmer_pca_labels,kmer_rules_pickle,kmer_rule_matrix):
    """Return dash data structures needed to run dash app. Copy them over to a directory to be referenced by app or uploaded to Heroku for dash app deployment."""
    from sklearn.preprocessing import LabelEncoder
    output_dir +='/'
    final_output_dir = output_dir + 'final_outputs/'
    if reference_genome != 'no_test':
        reference_genome = os.path.abspath(reference_genome)
    else:
        reference_genome = ''
    for path in [output_dir,final_output_dir]:
        try:
            os.makedirs(path)
        except:
            pass
    df = pd.read_csv(kmer_rule_matrix,index_col=0)
    df.to_csv(final_output_dir+kmer_rule_matrix.rsplit('/',1)[-1])
    kmers = list(df.index)
    #FIXME concatenate/stack below
    t_data = np.load(kmer_pca_data)
    if t_data.shape[1] > 3:
        t_data = KernelPCA(n_components=3).fit_transform(t_data)
    #print map(lambda x: np.array(x).shape,[list(df.index),pickle.load(open(kmer_pca_labels,'rb')),pickle.load(open(kmer_rules_pickle,'rb')),np.load(kmer_pca_data)])
    pca_data_df = pd.DataFrame(np.hstack([np.array(list(df.index))[:,np.newaxis],np.array(pickle.load(open(kmer_pca_labels,'rb')))[:,np.newaxis],np.array(pickle.load(open(kmer_rules_pickle,'rb')))[:,np.newaxis],t_data]),columns=['kmers','label','rule','x','y','z'])
    print pca_data_df
    pca_data_df.to_csv(final_output_dir+'pca_data.csv')
    if reference_genome:
        if 1:
            rule_mapping = zip(kmers,pca_data_df['rule'].as_matrix().tolist())
            with open(output_dir+'kmer_rules.fa','w') as f:
                f.write('\n'.join(['>%s\n%s'%(kmer,kmer) for kmer,rule in rule_mapping if rule != 'rule_-1']))
            subprocess.call('samtools faidx %s'%(reference_genome),shell=True)
            ctx.invoke(blast_kmers,blast_mem=blast_mem,reference_genome=reference_genome,kmer_fasta=output_dir+'kmer_rules.fa', output_file=output_dir+'kmer_rules.sam')
            subprocess.call("awk -v OFS='\\t' '{ print $3, $4, $4 + 1, $1 }' %s > %s"%(output_dir+'kmer_rules.sam',output_dir+'kmer_rules.bed'),shell=True)
            subprocess.call('samtools faidx %s && cut -f 1-2 %s.fai > %s.genome'%(reference_genome,reference_genome,output_dir + '/genome'),shell=True)
            subprocess.call('bedtools makewindows -g %sgenome.genome -w %d > %swindows.bed'%(output_dir,split_length,output_dir),shell=True)
            BedTool('%swindows.bed'%output_dir).sort().intersect(BedTool(output_dir+'kmer_rules.bed').sort(),wa=True,wb=True).sort().merge(c=7,o='collapse',d=-1).saveas(output_dir+'kmer_rules_merged.bed')
        le = LabelEncoder()
        le.fit(kmers)
        df = pd.read_table(output_dir+'kmer_rules_merged.bed', names = ['chr','xi','xf','kmers'], dtype = {'chr':str,'xi':np.int,'xf':np.int,'rule':str})
        if os.path.exists(final_output_dir+'kmer.count.npz'):
            rule_count_mat = sps.load_npz(final_output_dir+'kmer.count.npz')
            print rule_count_mat
        else:
            rule_count_mat = sps.dok_matrix((df.shape[0],len(kmers)),dtype=np.int)
            for i,rule_dist in enumerate(df['kmers']):
                d = Counter(rule_dist.split(','))
                rule_count_mat[i,le.transform(d.keys())] = np.array(d.values())
            rule_count_mat = rule_count_mat.tocsr()
            print rule_count_mat
            sps.save_npz(final_output_dir+'kmer.count.npz',rule_count_mat)
        df['xi'] = (df['xi'] + df['xf'])/2
        df = df.drop(['xf','kmers'],axis=1)
        df.to_csv(final_output_dir+'kmer.count.coords.csv')
        df2 = pd.SparseDataFrame(rule_count_mat,columns=kmers,default_fill_value=0)#pd.SparseDataFrame(sps.hstack([sps.csr_matrix(df['xi'].as_matrix()).T,rule_count_mat]),index=df['chr'].as_matrix().tolist(),columns=['xi']+list(kmers),default_fill_value=0).reset_index().rename(columns={'index':'chr'})
        df2.insert(0,'chr',df['chr'].as_matrix())
        df2.insert(1,'xi',df['xi'].as_matrix())
        df2.to_pickle(final_output_dir+'sparse_kmer_count_matrix.p')


@utilities.command(name='compareSubgenomes_progenitors_v_extracted')
@click.option('-s', '--scaffolds_file', default='scaffolds.p', help='Path to scaffolds pickle file.', show_default=True, type=click.Path(exists=False))
@click.option('-b1', '--bed1', default='bootstrap_results.bed', help='Bed file containing bootstrapped results.', show_default=True, type=click.Path(exists=False))
@click.option('-b2', '--bed2', default='progenitors.bed', help='Bed file containing progenitor results. Alternatively, you can include the results from a prior bootstrap.', show_default=True, type=click.Path(exists=False))
@click.option('-od', '--out_dir', default='./', help='Write results to this output directory.', show_default=True, type=click.Path(exists=False))
def compareSubgenomes_progenitors_v_extracted(scaffolds_file, bed1, bed2, out_dir):
    """Compares the results found from the progenitorMapping function to results from polyCRACKER. Outputs CompareSubgenomesToProgenitors* files in the output directory. Final_stats is a better alternative to this function."""
    import seaborn as sns
    if out_dir.endswith('/') == 0:
        out_dir += '/'
    scaffolds = pickle.load(open(scaffolds_file,'rb'))
    scaffoldsDict = {scaffold: '\t'.join(['_'.join(scaffold.split('_')[0:-2])] + scaffold.split('_')[-2:]) for
                     scaffold in scaffolds}
    scaffoldsBed = BedTool('\n'.join(scaffoldsDict.values()), from_string=True)
    subgenomesDicts = []
    for bed_file in [bed1,bed2]:
        featureBed = BedTool(bed_file)
        finalBed = scaffoldsBed.intersect(featureBed, wa=True, wb=True).sort().merge(d=-1, c=7, o='distinct')
        scaffolds_final = defaultdict(list)
        for line in str(finalBed).splitlines():
            lineList = line.strip('\n').split('\t')
            feature = lineList[-1]
            scaffold = '_'.join(lineList[0:-1])
            if ',' not in feature:
                scaffolds_final[feature].append(scaffold)
        for feature in scaffolds_final.keys():
            scaffolds_final[feature] = set(scaffolds_final[feature])
        subgenomesDicts.append(scaffolds_final)
    finalDict = defaultdict(list)
    for key in subgenomesDicts[0]:
        finalDict[key] = {key2: len(subgenomesDicts[0][key].intersection(subgenomesDicts[1][key2])) for key2 in subgenomesDicts[1]}
    df = pd.DataFrame(finalDict)
    correlation = compute_correlation(df.as_matrix())
    df.to_csv(out_dir + 'CompareSubgenomesToProgenitors.csv')
    plt.figure()
    sns_plot = sns.heatmap(df, annot=True)
    plt.title('Extracted Subgenomes v Progenitors, r = %.4f'%(correlation))
    plt.xlabel(bed1 + ', collected %.2f %% scaffolds'%(sum([len(subgenomesDicts[0][key]) for key in subgenomesDicts[0].keys()])/float(len(scaffolds))*100))
    plt.ylabel(bed2 + ', collected %.2f %% scaffolds'%(sum([len(subgenomesDicts[1][key]) for key in subgenomesDicts[1].keys()])/float(len(scaffolds))*100))
    plt.savefig(out_dir + 'CompareSubgenomesToProgenitors.png', dpi=300)
    plt.savefig(out_dir + 'CompareSubgenomesToProgenitors.pdf', dpi=300)


@utilities.command(name='plot_rules_chromosomes')
@click.pass_context
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory for computations of spatial plotting of kmer rules.', type=click.Path(exists=False))
@click.option('-go', '--original_genome', help='Complete path to original, prechunked genome fasta', type=click.Path(exists=False))
@click.option('-csv', '--kmer_rule_matrix', default = './kmer_master_count_chi2_matrix_rules.csv', show_default = True, help='Kmer count matrix with appended rule labels. Default includes chi-squared values.', type=click.Path(exists=False))
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
@click.option('-sl', '--split_length', default=75000, help='Length of intervals, bed files.', show_default=True)
def plot_rules_chromosomes(ctx,work_dir, original_genome, kmer_rule_matrix, blast_mem, split_length):
    """Plot distribution of rules/conservation pattern kmers across the chromosomes."""
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    # subset out kmers by rules >rule\nkmer
    original_genome = os.path.abspath(original_genome)
    subprocess.call('samtools faidx %s'%original_genome,shell=True)
    original_genome = original_genome.rsplit('/',1)
    if original_genome[0:-1]:
        fasta_path = original_genome[0] +'/'
    else:
        fasta_path = './'
    original_genome = original_genome[-1]
    work_dir += '/' if work_dir.endswith('/') else ''
    blast_path, bed_path, sort_path = tuple([work_dir+path for path in ['rule_blast/','rule_bed/', 'rule_sort/']])
    for path in [work_dir,blast_path, bed_path, sort_path]:
        try:
            os.makedirs(path)
        except:
            pass
    df = pd.read_csv(kmer_rule_matrix,index_col=0)
    kmers = list(df.index)
    if 1:
        rule_mapping = zip(kmers,np.vectorize(lambda x: x.replace(' ','_'))(df['Rules']))
        with open(blast_path+'kmer_rules.higher.kmers.fa','w') as f:
            f.write('\n'.join(['>%s\n%s'%(rule,kmer) for kmer,rule in rule_mapping if rule != 'rule_-1']))
        # blast these kmers across genome
        writeBlast(original_genome, blast_path, blast_path, fasta_path, 1, blast_mem)
        #ctx.invoke(blast2bed,blast_file=blast_path+'kmer_rules.higher.kmers.sam', bb=1 , low_memory=0, output_bed_file=work_dir+bed_path+'kmer_rules.bed', output_merged_bed_file=work_dir+bed_path+'kmer_rules_merged.bed', external_call=1)
        subprocess.call("awk -v OFS='\\t' '{ print $3, $4, $4 + 1, $1 }' %s > %s"%(blast_path+'kmer_rules.higher.kmers.sam',bed_path+'kmer_rules.bed'),shell=True)
        subprocess.call('cut -f 1-2 %s.fai > %s.genome'%(fasta_path+original_genome,work_dir + '/genome'),shell=True)
        subprocess.call('bedtools makewindows -g %sgenome.genome -w %d > %swindows.bed'%(work_dir,split_length,work_dir),shell=True)
        BedTool('%swindows.bed'%work_dir).sort().intersect(BedTool(bed_path+'kmer_rules.bed').sort(),wa=True,wb=True).sort().merge(c=7,o='collapse',d=-1).saveas(bed_path+'kmer_rules_merged.bed')
    # FIXME intersect with windows bed and merge with d=-1
    # FIXME then read last column and encode labels into matrix, to append to dataframe
    #FIXME create windows bed
    #blast2bed3(work_dir, blast_path, bed_path, sort_path, original_genome, 1, bedgraph=0)
    #ctx.invoke(generate_unionbed, bed3_directory=bed_path, original=1, split_length=split_length, fai_file=fasta_path+original_genome+'.fai', work_folder=work_dir)
    # plot results, one chromosome at time
    le = LabelEncoder()
    rules = [rule.replace(' ','_') for rule in sorted(df['Rules'].unique().tolist()) if rule != 'rule -1']
    #print rules
    le.fit(rules)
    df = pd.read_table(bed_path+'kmer_rules_merged.bed', names = ['chr','xi','xf','rule'], dtype = {'chr':str,'xi':np.int,'xf':np.int,'rule':str})
    #print df
    rule_count_mat = sps.dok_matrix((df.shape[0],len(rules)),dtype=np.int)
    for i,rule_dist in enumerate(df['rule']):
        d = Counter(rule_dist.split(','))
        rule_count_mat[i,le.transform(d.keys())] = np.array(d.values())

    df['xi'] = (df['xi'] + df['xf'])/2
    df = df.drop(['xf','rule'],axis=1)
    df = pd.concat([df,pd.SparseDataFrame(rule_count_mat,columns=rules,default_fill_value=0).to_dense()],axis=1)

    """
    df = pd.read_table(work_dir+'subgenomes.union.bedgraph', names = ['chr','xi','xf']+[rule for rule in rules if rule != 'rule -1'])
    #df = df.rename(columns=['chr','xi','xf']+[rule for rule in df['Rules'].unique.as_matrix().tolist() if rule != 'rule -1'])
    df.iloc[:,0:3] = np.array(map(lambda x: x.rsplit('_',2), df['chr'].as_matrix().tolist()))
    df.iloc[:,1:] = df.iloc[:,1:].as_matrix().astype(np.int)
    df['xi'] = (df['xi'] + df['xf'])/2
    df = df.drop(['xf'],axis=1)

    df = df.sort_values(['chr','xi'])
    # melt the dataframe and then use sns tsplot to plot all of these values"""
    #print df
    df = pd.melt(df, id_vars=['chr','xi'], value_vars=rules)#[rule for rule in rules if rule != 'rule -1'])
    #print df
    df = df.rename(columns = {'variable':'Rule','value':'Number_Hits','xi':'x_mean'})
    df['subject'] = 0
    #print df
    for chromosome in set(df['chr'].as_matrix()):
        print df[df['chr'].as_matrix()==chromosome]
        plt.figure()
        sns.tsplot(data=df[df['chr']==chromosome],condition='Rule',unit='subject',interpolate=True,legend=True,value='Number_Hits',time='x_mean')
        plt.savefig(work_dir+chromosome+'.png',dpi=300)


@utilities.command(name='merge_split_kmer_clusters')
@click.pass_context
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for merging of clusters is done.', type=click.Path(exists=False))
@click.option('-c', '--estimated_clusters', default=-1, show_default=True, help='Estimated number of clusters based on prior rule discovery. If set to -1, bases number of cluster selection on number already discovered.')
@click.option('-d', '--n_dimensions', default=3, help='Number of dimensions to reduce to.', show_default=True)
def merge_split_kmer_clusters(ctx,work_dir,estimated_clusters,n_dimensions):
    """In development: working on merging and splitting clusters of particular kmer conservation patterns acros multiple genomes."""
    import statsmodels.api as sm
    from scipy.stats import pearsonr
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    def linear_test(x, y, s_x, s_y):
        # must have high positive r value, reduced chi-square close to one to merge
        # expected equation is y=x for clusters to be similar, find if slope = 1 and intercept = 0
        print x,y,s_x,s_y
        R_mat = np.ones((2,2))/np.sqrt(2)
        R_mat[1,0] = -R_mat[1,0]
        X = np.vstack((x,y))
        sX = np.vstack((s_x,s_y))
        X = np.dot(R_mat,X)
        sX = np.dot(R_mat,sX)
        results_y = sm.WLS(X[1,:],sm.add_constant(X[0,:]),weights=1./sX[1,:]).fit()
        results_x = sm.WLS(X[0,:],sm.add_constant(X[1,:]),weights=1./sX[0,:]).fit()
        print(results_x.summary())
        print(results_y.summary())
        #f1 = np.polyval(np.polyfit(x,y,deg=1),x) # s_y
        #f2 = np.polyval(np.polyfit(y,x,deg=1),y) # s_x
        #chi2_y_reduced = (f1-y)**2/s_y/(len(y) - 2)
        #chi2_x_reduced = (f2-x)**2/s_x/(len(x) - 2)
        conditional = np.all(np.hstack((results_y.pvalues,results_x.pvalues)) > 0.05)
        print conditional
        return conditional
    rules_df = pd.read_csv(work_dir+'kmer_master_count_chi2_matrix_rules.csv', index_col=0)
    df = rules_df.iloc[:,:-1]
    df_train = df.iloc[rules_df['Rules'] != 'rule -1',:]
    df_test = df.iloc[rules_df['Rules'] == 'rule -1',:]
    rules_mean = pd.read_csv(work_dir+'kmers_unsupervised_rules_chi2.csv')
    rules_std = pd.read_csv(work_dir+'kmers_unsupervised_rules_chi2_uncertainty.csv')
    X_mean = rules_mean.to_matrix()
    X_std = rules_std.to_matrix()
    model = Pipeline([('scaler',StandardScaler()),('lda',LDA(n_components=n_dimensions))])
    model.fit(df_train)
    t_data = model.transform(df)
    np.save(work_dir+'LDA_kmers.npy',t_data)
    ctx.invoke(plotPositions,positions_npy=work_dir+'LDA_kmers.npy',labels_pickle=work_dir+'kmer_labels_coordinates.p',colors_pickle=work_dir+'rules_chi2.p',output_fname=work_dir+'subgenome_bio_rules_kmers_chi2.html')
    # combine rules
    merged_clusters = []
    for i,j in combinations(range(len(set(rules_df['Rules'].to_matrix()))), 2): # labels.max()+1
        if pearsonr(X_mean[i,:],X_mean[j,:]) > 0.7 and linear_test(X_mean[i,:], X_mean[j,:], X_std[i,:], X_std[j,:]):
            merged_clusters.append((i,j))
    #labels = BayesianGaussianMixture(n_components=(len(set(rules_df['Rules'].to_matrix())) if estimated_clusters == -1 else estimated_clusters))
    print 'MERGED: ', merged_clusters


@utilities.command()
@click.option('-c', '--input_labels', default='colors_pickle.p', help='Pickle file containing the cluster/class each label/scaffold belongs to.', show_default=True, type=click.Path(exists=False))
@click.option('-s', '--scaffolds_pickle', default='scaffolds.p', help='Path to scaffolds pickle file.', show_default=True, type=click.Path(exists=False))
@click.option('-o', '--output_pickle', default='labels_new.p', help='Pickle file containing the new cluster/class of scaffolds.', show_default=True, type=click.Path(exists=False))
@click.option('-npy', '--transformed_data', default='', help='Input transformed matrix. Can leave empty, but if using this, label propagater will calculate distances from this matrix. Recommended use: differential kmer or TE matrix or PCA reduced data.', type=click.Path(exists=False))
@click.option('-r', '--radius', default=100000., help='Total radius in bps of search to look for neighboring classified sequences for RadiusNeighborsClassifier.', show_default=True)
def genomic_label_propagation(input_labels,scaffolds_pickle,output_pickle, transformed_data,radius):
    """Extend polyCRACKER labels up and downstream of found labels. Only works on split scaffold naming."""
    from sklearn.neighbors import RadiusNeighborsClassifier
    scaffolds = pickle.load(open(scaffolds_pickle,'rb'))
    initial_labels = pickle.load(open(input_labels,'rb'))
    labels_dict = {label: i for i, label in enumerate(np.setdiff1d(np.unique(initial_labels),['ambiguous']).tolist())}
    labels_dict['ambiguous'] = -1
    inverse_dict = {i:label for label,i in labels_dict.items()}
    codified_labels = np.vectorize(lambda x: labels_dict[x])(initial_labels)
    print codified_labels
    if transformed_data:
        genomic_space = np.load(transformed_data)
    else:
        genomic_space = find_genomic_space(scaffolds)
    #label_prop = LabelSpreading(kernel='knn',n_neighbors=n_neighbors)
    #label_prop.fit(genomic_space,codified_labels)
    radius = radius
    r_neigh = RadiusNeighborsClassifier(radius = radius, outlier_label=-1)
    r_neigh.fit(genomic_space[codified_labels != -1],codified_labels[codified_labels != -1])
    propagated_labels = codified_labels
    propagated_labels[propagated_labels == -1] = r_neigh.predict(genomic_space[codified_labels == -1])
    #print neigh.kneighbors_graph(genomic_space,n_neighbors=15).todense()
    new_labels = np.vectorize(lambda x: inverse_dict[x])(propagated_labels)#(label_prop.predict(genomic_space))
    pickle.dump(new_labels,open(output_pickle,'wb'))
    with open(output_pickle.replace('.p','.bed'),'w') as f:
        for scaffold, label in zip(scaffolds.tolist(),new_labels.tolist()):
            sl = scaffold.split('_')
            f.write('\t'.join(['_'.join(sl[:-2])]+sl[-2:]+[label])+'\n')


@utilities.command(name='unionbed2matrix')
@click.option('-u', '--unionbed', default='subgenomes.union.bedgraph', help='Unionbed file containing counts of differential kmers or TEs.', show_default=True, type=click.Path(exists=False))
@click.option('-o', '--output_matrix', default='subgenomes.union.bed_matrix.npy', help='Output path to differential kmer/TE matrix', show_default=True, type=click.Path(exists=False))
def unionbed2matrix(unionbed, output_matrix):
    """Convert unionbed file into a matrix of total differential kmer counts for each scaffold."""
    with open(unionbed,'r') as f:
        out_matrix = np.array([map(float,line.split()[3:]) for line in f if line])
    np.save(output_matrix,out_matrix)


@utilities.command(name='generate_out_bed')
@click.option('-id', '--input_dir', default='./', help='Directory containing solely the cluster/classified outputs, with names stored in txt files.', show_default=True, type=click.Path(exists=False))
@click.option('-o', '--original', default=0, help='Select 1 if subgenomeExtraction was done based on original nonchunked genome.', show_default=True)
@click.option('-fai', '--fai_file', help='Full path to original, nonchunked fai file.', type=click.Path(exists=False))
@click.option('-fo', '--output_fname', default = 'output.bed', help='Desired output bed file.', show_default=True, type=click.Path(exists=False))
def generate_out_bed(input_dir, original, fai_file, output_fname):
    """Find cluster labels for all clusters/subgenomes in directory, output to bed."""
    try:
        original = int(original)
    except:
        original = 0
    if input_dir.endswith('/') == 0:
        input_dir += '/'
    open(output_fname,'w').close()
    if original:
        genomeDict = {line.split('\t')[0]: line.split('\t')[1] for line in open(fai_file, 'r') if line}
    for file in os.listdir(input_dir):
        if file.endswith('.txt') and file.startswith('subgenome_'):
            subgenome = file.split('.')[0]
            if original == 0:
                with open(input_dir+file,'r') as f ,open(output_fname,'a') as f2:
                    f2.write('\n'.join(['\t'.join(['_'.join(line.strip('\n').split('_')[0:-2])] + line.strip('\n').split('_')[-2:] + [subgenome]) for line in f if line])+'\n')
            else:
                with open(input_dir + file, 'r') as f, open(output_fname, 'a') as f2:
                    f2.write('\n'.join(['\t'.join([line.strip('\n').split('\t')[0],'0',genomeDict[line.strip('\n').split('\t')[0]]] + [subgenome]) for
                                        line in f if line]) + '\n')


@utilities.command(name='progenitorMapping')
@click.pass_context
@click.option('-i', '--input_fasta', help='Complete path to input polyploid fasta file, original or chunked.', type=click.Path(exists=False))
@click.option('-p', '--progenitor_fasta_folder', default='./progenitors/', help='Folder containing progenitor genomes.', show_default=True, type=click.Path(exists=False))
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
@click.option('-o', '--original', default=0, help='Select 1 if trying to extract subgenomes based on original nonchunked genome.', show_default=True)
@click.option('-kl', '--kmer_length', default='23', help='Length of kmers to use for subgenome extraction.', show_default=True)
@click.option('-fout', '--outputfname', default='progenitors.bed', help='Output progenitor name', show_default=True, type=click.Path(exists=False))
@click.option('-pf', '--progenitor_fastas', default = '', help='Optional: in addition to the progenitor fasta folder, directly specify the filename of each progenitor fasta and how you would like each fasta to be labelled as. Comma delimited list of fasta:label pairing. Example usage: -pf subgenomeT.fa:T,subgenomeA.fa:A,subgenomeR.fa:R', type=click.Path(exists=False))
@click.option('-gff', '--gff_file', default = '', help='Optional:If progenitor_fastas specified, you have the option to input gff file, which will be split up into progenitor specific gene bed files for future synteny run.',show_default=True)
@click.option('-go', '--original_genome', default='', show_default=True, help='If running synteny: Filename of polyploid fasta; must be full path to original. Use if input gff file. Need to install jcvi library to run', type=click.Path(exists=False))
def progenitorMapping(ctx,input_fasta, progenitor_fasta_folder, blast_mem, original, kmer_length, outputfname, progenitor_fastas, gff_file, original_genome):
    """Takes reference progenitor fasta files, and bins an input polyploid file according to progenitor. These results will be compared to polyCRACKER's results."""
    create_path(progenitor_fasta_folder)
    genome_info = []
    if progenitor_fasta_folder.endswith('/') == 0:
        progenitor_fasta_folder += '/'
    seal_outdir = progenitor_fasta_folder + 'seal_outputs/'
    create_path(seal_outdir)
    if progenitor_fastas:
        progenitor_fastas = dict([tuple(progenitor.split(':')) for progenitor in progenitor_fastas.split(',')])
        progenitorFastas = ','.join([progenitor_fasta_folder+progenitor_fasta for progenitor_fasta in progenitor_fastas])
        progenitor_labels = {progenitor_fasta.replace('.fasta','').replace('.fa',''):label for progenitor_fasta,label in progenitor_fastas.items()}
    else:
        progenitorFastas = ','.join([progenitor_fasta_folder + file for file in os.listdir(progenitor_fasta_folder) if file.endswith('.fasta') or file.endswith('.fa')])
    subprocess.call('seal.sh ref=%s in=%s pattern=%sout_%%.seal.fasta outu=%s/unmatched.seal.ignore.fasta ambig=all '
                    'refnames overwrite=t k=%s refstats=match.stats threads=4 -Xmx%sg'%(progenitorFastas,input_fasta,seal_outdir,seal_outdir,kmer_length,blast_mem),shell=True)
    subgenomes = []
    if original:
        genomeDict = {line.split('\t')[0]: line.split('\t')[1] for line in open(input_fasta+'.fai', 'r') if line}
    open(outputfname,'w').close()
    count = 0
    for file in os.listdir(seal_outdir):
        if file.endswith('seal.fasta'):
            if progenitor_fastas:
                subgenomeName = 'Subgenome_' + progenitor_labels[file.replace('out_','').replace('.seal.fasta','')]
            else:
                subgenomeName = 'Subgenome_' + chr(65+count)
            subgenomes.append(file)
            Fasta(seal_outdir+file)
            if original:
                with open(seal_outdir + file + '.fai','r') as f, open(outputfname,'a') as f2:
                    f2.write('\n'.join(['\t'.join([line.strip('\n').split('\t')[0],'0',genomeDict[line.strip('\n').split('\t')[0]]] + [subgenomeName]) for line in f if line])+'\n')
            else:
                with open(seal_outdir + file + '.fai','r') as f, open(outputfname,'a') as f2:
                    f2.write('\n'.join(['\t'.join(['_'.join(line.strip('\n').split('\t')[0].split('_')[0:-2])] + line.strip('\n').split('\t')[0].split('_')[-2:] + [subgenomeName]) for line in f if line])+'\n')
            count += 1
    if progenitor_fastas:
        for progenitor_label in progenitor_labels.values():
            subprocess.call('grep Subgenome_%s progenitors.bed | bedtools sort | bedtools merge > progenitor_%s.bed'%(progenitor_label,progenitor_label),shell=True)
        if gff_file:
            gff = BedTool(gff_file)
            for progenitor_label in progenitor_labels.values():
                progenitor_bed = BedTool('progenitor_%s.bed'%progenitor_label)
                progenitor_gff = 'progenitor_%s.gff3'%progenitor_label
                gff.intersect(progenitor_bed,wa=True).sort().saveas(progenitor_gff)
                with open('progenitor_%s.bed'%progenitor_label,'w') as f:
                    for line in str(progenitor_bed).splitlines():
                        if line:
                            f.write(line + '\t%s\n'%('_'.join(line.split())))
                subprocess.call('python -m jcvi.formats.gff bed --type=mRNA --key=gene_name %s > progenitor%s.bed'%(progenitor_gff,progenitor_label),shell=True)
                if original_genome:
                    subprocess.call('samtools faidx '+original_genome,shell = True)
                    subprocess.call('python -m jcvi.formats.gff load %s %s --parents=mRNA --children=CDS -o progenitor%s.cds'%(progenitor_gff,original_genome,progenitor_label),shell=True)
            if original_genome and gff_file:
                for proj1, proj2 in combinations(progenitor_labels.values(), 2):
                    subprocess.call('python -m jcvi.compara.catalog ortholog progenitor%s progenitor%s'%(proj1, proj2),shell=True)
                    ctx.invoke(anchor2bed,anchor_file='protenitor%s.progenitor%s.lifted.anchors'%(proj1,proj2),qbed='progenitor%s.bed'%proj1, sbed='progenitor%s.bed'%proj2)
                    #subprocess.call('python -m jcvi.assembly.syntenypath bed %s --switch --scale=10000 --qbed=progenitor%s.bed --sbed=progenitor%s.bed -o synteny.%s.%s.bed'%(proj1+'_'+proj2+'.bed',proj1,proj2,proj1,proj2),shell=True)


@utilities.command(name='align')
@click.pass_context
@click.option('-f1', '--fasta1', default='', show_default=True, help="Input fasta one. Comma delimit and set fasta2 to '' to build alignment matrix.", type=click.Path(exists=False))
@click.option('-f2', '--fasta2', default='', show_default=True, help='Input fasta two.', type=click.Path(exists=False))
@click.option('-b', '--both', is_flag=True, help='Perform 2 alignments by switching the order of the two genomes.')
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory for final outputs.', type=click.Path(exists=False))
@click.option('-bed', '--to_bed', is_flag=True, help='Convert MAF output to bed and compute identity.')
def align(ctx, fasta1,fasta2, both, work_dir, to_bed):
    """Align two fasta files."""
    work_dir += '/'
    if not fasta2:
        fastas = fasta1.split(',')
        similarity_matrix = pd.DataFrame(np.ones((len(fastas),len(fastas))).astype(float),index=fastas,columns=fastas)
        flagged = 1
    else:
        fastas = [(fasta1,fasta2)]
        flagged = 0
    for fasta1, fasta2 in combinations(fastas,r=2):
        last_files = []
        subprocess.call('python -m jcvi.apps.align last %s %s --format MAF'%(fasta1,fasta2),shell=True)
        last = fasta2[fasta2.rfind('/')+1:fasta2.rfind('.')] + '.' + fasta1[fasta1.rfind('/')+1:fasta1.rfind('.')] + '.last'
        shutil.copy(last, work_dir)
        last_files.append(work_dir+last)
        if both:
            subprocess.call('python -m jcvi.apps.align last %s %s --format MAF'%(fasta2,fasta1),shell=True)
            last = fasta1[fasta1.rfind('/')+1:fasta1.rfind('.')] + '.' + fasta2[fasta2.rfind('/')+1:fasta2.rfind('.')] + '.last'
            shutil.copy(last, work_dir)
            last_files.append(work_dir+last)
        if to_bed:
            ctx.invoke(maf2bed,last=','.join(last_files),work_dir=work_dir)
        if to_bed and flagged:
            with open(work_dir+'weighted_sum.txt','r') as f:
                similarity = float(f.read().splitlines()[0])
            similarity_matrix.loc[fasta1,fasta2] = similarity
            similarity_matrix.loc[fasta2,fasta1] = similarity
    if to_bed and flagged:
        similarity_matrix.to_csv(work_dir+'similarity_matrix.csv')


if __name__ == '__main__':
    utilities()
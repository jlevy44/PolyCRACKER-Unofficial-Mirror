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
def repeats():
    pass


def create_path(path):
    """Create a path if directory does not exist, raise exception for other errors"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def write_informative_diff_kmers(informative_diff_kmers, work_dir):
    kmer_dir = work_dir+'informative_diff_kmers/'
    try:
        os.makedirs(kmer_dir)
    except:
        pass
    for subgenome in informative_diff_kmers:
        with open(kmer_dir+subgenome+'informative.higher.kmers.fa','w') as f:
            f.write('\n'.join(['>%s\n%s'%(kmer,kmer) for kmer in informative_diff_kmers[subgenome]]))
    return kmer_dir


def jaccard_similarity(sparse_matrix):
    """similarity=(ab)/(aa + bb - ab)"""
    cols_sum = sparse_matrix.getnnz(axis=0)
    ab = sparse_matrix.T * sparse_matrix
    aa = np.repeat(cols_sum, ab.getnnz(axis=0))
    bb = cols_sum[ab.indices]
    similarities = ab.copy()
    similarities.data /= (aa + bb - ab.data)
    return similarities


def output_Dendrogram(distance_matrix_npy, kmers_pickle, out_dir):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    import plotly.figure_factory as ff
    distance_matrix = np.load(distance_matrix_npy)
    dists = squareform(distance_matrix)
    linkage_mat = linkage(dists, 'single')
    kmers = pickle.load(open(kmers_pickle,'rb'))
    plt.figure()
    dendrogram(linkage_mat,labels=kmers)
    plt.savefig(out_dir+'output_dendrogram.png')
    fig = ff.create_dendrogram(linkage_mat, orientation='left', labels=kmers)
    fig['layout'].update({'width':1200, 'height':1800})
    py.plot(fig, filename=out_dir+'output_dendrogram.html', auto_open=False)


def repeatGFF2Bed(repeat_gff, out_dir, labels = 0):
    if repeat_gff.endswith('.gff') or repeat_gff.endswith('.gff3'):
        output_bed = out_dir+'repeats.bed'
        repeat_class_subclass = []
        repeat_motifs = []
        BedTool(repeat_gff).sort().saveas(repeat_gff)
        #subprocess.call('bedtools sort -i %s > temp.gff3 && mv temp.gff3 > %s'%(repeat_gff,repeat_gff), shell=True)
        with open(repeat_gff,'r') as f:
            repeat_elements = []
            repeat_elements2 = []
            for line in f:
                if line and line.startswith('#') == 0:
                    lineList = line.strip('\n').split()
                    subclass = lineList[-1].strip('"')
                    motif = lineList[-4].strip('"').replace('Motif:','')
                    repeat_class_subclass.append(subclass)
                    repeat_motifs.append(motif)
                    repeat_elements.append('%s\t%d\t%s'%(lineList[0],int(lineList[3])-1,lineList[4]))
                    repeat_elements2.append(motif+'#'+subclass)


        repeat_elements, idxs = np.unique(np.array(repeat_elements),return_index=True)
        repeat_elements2 = np.array(repeat_elements2)[idxs]
        repeat_class_subclass = np.array(repeat_class_subclass)[idxs]
        repeat_motifs = np.array(repeat_motifs)[idxs]

        if labels:
            with open(output_bed, 'w') as f2:
                f2.write('\n'.join(np.vectorize(lambda i: '%s\t%s'%(repeat_elements[i],repeat_elements2[i]))(range(len(repeat_elements)))))
        else:
            with open(output_bed, 'w') as f2:
                f2.write('\n'.join(repeat_elements))

        pickle.dump(repeat_elements2,open(out_dir + 'TE_motif_subclass_raw_list.p','wb'))
        repeat_elements2 = Counter(repeat_elements2)

        repeat_elements = np.vectorize(lambda x: x.replace('\t','_'))(repeat_elements)

        pickle.dump(repeat_elements,open(out_dir + 'repeat_elements.p','wb'))
        pickle.dump(repeat_elements2.keys(),open(out_dir + 'TE_motif_subclass.p','wb'))
        pickle.dump(repeat_elements2,open(out_dir + 'TE_motif_subclass_counter.p','wb'))
        pickle.dump(repeat_motifs,open(out_dir + 'repeat_motifs.p','wb'))
        pickle.dump(repeat_class_subclass,open(out_dir + 'repeat_class_subclass.p','wb'))
        pickle.dump({repeat_elements[i]:repeat_class_subclass[i] for i in range(len(repeat_elements))}, open(out_dir + 'repeat_class_dictionary.p','wb'))
    elif repeat_gff.endswith('.bed'):
        output_bed = repeat_gff
    else:
        output_bed = 'xxx'
    return output_bed


def sam2diffkmer_clusteringmatrix(work_dir,kmers_pickle,blast_path, kernel, repeat_bed, TE_cluster = 0):
    if TE_cluster:
        windows = BedTool(work_dir + 'windows.bed')
        kmers = pickle.load(open(kmers_pickle,'rb'))
        with open(work_dir + 'windows.bed','r') as f:
            scaffolds = np.array([line.replace('\t','_') for line in f.read().splitlines() if line])
        pickle.dump(scaffolds, open(work_dir+'scaffolds_TE_cluster_analysis.p','wb'))
        repeat_masked_TEs = windows.intersect(repeat_bed,wa=True,wb=True).sort()
        TE_repeat_masked = ''
        for line in str(repeat_masked_TEs).splitlines():
            if line:
                lineList = line.strip('\n').split()
                TE_repeat_masked += '\t'.join(['_'.join(lineList[0:3]), '0', str(int(lineList[2]) - int(lineList[1])), lineList[-1]]) + '\n'
        TE_repeat_masked = BedTool(TE_repeat_masked,from_string=True).sort().merge(c=4, o='collapse').sort()
        bedText = []
        for line in str(TE_repeat_masked).splitlines():
            if line:
                lineList = line.strip('\n').split()
                lineList2 = lineList[0].split('_')
                lineList[1:-1] = map(lambda x: str(int(lineList2[-2])+ int(x.strip('\n'))), lineList[1:-1])
                lineList[0] = '_'.join(lineList2[:-2])
                bedText.append('\t'.join(lineList))
        BedTool('\n'.join(bedText),from_string=True).sort().saveas(work_dir+'clustering_TEs.bed')
        build_cluster_matrix = [(scaffolds, work_dir+'clustering_TEs.bed',work_dir+'TEclusteringMatrix.npz',work_dir+'TE_cluster_pca_data.npy')]
    else:
        # turn sam files into clustering matrix by scaffold
        repeat_analysis = 0
        if repeat_bed.endswith('.bed'):
            repeat_folder = work_dir + 'repeat_analysis/'
            repeat_analysis = 1
            repeat_windows = BedTool(repeat_bed)
        kmers = pickle.load(open(kmers_pickle,'rb'))
        with open(work_dir + 'windows.bed','r') as f:
            scaffolds = np.array([line.replace('\t','_') for line in f.read().splitlines() if line])
        pickle.dump(scaffolds, open(work_dir+'scaffolds_diff_kmer_analysis.p','wb'))
        windows = BedTool(work_dir + 'windows.bed')
        kmers_in_windows = []
        kmers_in_repeat_masked = []
        """for file in os.listdir(blast_path):
            if file.endswith('.sam'):
                #subprocess.call("sam2bed < %s | awk -v OFS='\\t' '{print $1, $2, $3, $4}' > %stemp.bed"%(blast_path+file, work_dir),shell=True)
                query = BedTool(work_dir+'temp.bed').sort()
                kmers_in_windows.append(windows.intersect(query,wa=True,wb=True).merge(c=7,o='collapse'))
                if repeat_analysis:
                    kmers_in_repeat_masked.append(repeat_windows.intersect(query,wa=True,wb=True))#.merge(c=7,o='collapse')
        kmers_in_windows[0].cat(*kmers_in_windows[1:],c=4,o='collapse').saveas(work_dir+'clustering.bed')"""
        subprocess.call("cat %s > %stemp.sam && sam2bed < %stemp.sam | awk -v OFS='\\t' '{print $1, $2, $3, $4}' > %stemp.bed"%(' '.join([blast_path+file for file in os.listdir(blast_path) if file.endswith('.sam')]), work_dir, work_dir, work_dir),shell=True)
        query = BedTool(work_dir+'temp.bed').sort()
        windows.intersect(query,wa=True,wb=True).merge(c=7,o='collapse',d=-1).saveas(work_dir+'clustering.bed')
        build_cluster_matrix = [(scaffolds,work_dir+'clustering.bed',work_dir+'diff_kmer_sparse_matrix.npz',work_dir+'transformed_diff_kmer_matrix.npy')]
        if repeat_analysis:
            #repeat_masked_kmers = kmers_in_repeat_masked[0].cat(*kmers_in_repeat_masked[1:],postmerge=False).sort()#,c=4,o='collapse'
            repeat_masked_kmers = repeat_windows.intersect(query,wa=True,wb=True).sort()
            print repeat_masked_kmers.head()
            kmer_repeat_masked = ''
            for line in str(repeat_masked_kmers).splitlines():
                if line:
                    lineList = line.strip('\n').split()
                    kmer_repeat_masked += '\t'.join(['_'.join(lineList[0:3]), '0', str(int(lineList[2]) - int(lineList[1])), lineList[-1]]) + '\n'
            kmer_repeat_masked = BedTool(kmer_repeat_masked,from_string=True).sort().merge(c=4, o='collapse').sort()
            print kmer_repeat_masked.head()
            bedText = []
            for line in str(kmer_repeat_masked).splitlines():
                if line:
                    lineList = line.strip('\n').split()
                    lineList2 = lineList[0].split('_')
                    lineList[1:-1] = map(lambda x: str(int(lineList2[-2])+ int(x.strip('\n'))), lineList[1:-1])
                    lineList[0] = '_'.join(lineList2[:-2])
                    bedText.append('\t'.join(lineList))
            BedTool('\n'.join(bedText),from_string=True).sort().saveas(repeat_folder+'clustering_repeats.bed')
            build_cluster_matrix.append((pickle.load(open(repeat_folder+'repeat_elements.p','rb')), repeat_folder+'clustering_repeats.bed',repeat_folder+'TE_sparse_matrix.npz',repeat_folder+'transformed_TE_matrix.npy'))
    kmer_index = {kmer: i for i, kmer in enumerate(kmers)}
    for scaffolds_new, clusterBed, output_sparse, output_positions in build_cluster_matrix:
        scaffolds_index = {scaffold : i for i, scaffold in enumerate(scaffolds_new)}
        data = sps.dok_matrix((len(scaffolds_new), len(kmers)),dtype=np.float32)
        with open(clusterBed, 'r') as f:
            """
            if clusterBed.endswith('clustering_repeats.bed'):
                for line in f:
                    if line:
                        try:
                            lineL = line.strip('\n').split('\t')
                            data[scaffolds_index['_'.join(lineL[0:3])], kmer_index[lineL[-1]]] += 1.
                        except:
                            pass
            else:
            """
            for line in f:
                if line:
                    listLine = line.rstrip('\n').split('\t')
                    if '_'.join(listLine[0:3]) in scaffolds_new:
                        counts = Counter(listLine[-1].split(','))
                        for key in counts:
                            try:
                                data[scaffolds_index['_'.join(listLine[0:3])], kmer_index[key]] = counts[key]
                            except:
                                pass
                    else: #FIXME test
                        print '_'.join(listLine[0:3])
        data = data.tocsc()
        sps.save_npz(output_sparse,data)
        data = data.tocsr()
        data = StandardScaler(with_mean=False, copy=False).fit_transform(data)
        try:
            if clusterBed.endswith('clustering_repeats.bed') == 0:
                transformed_data = KernelPCA(n_components=3,kernel=kernel, copy_X=False).fit_transform(data)
            else:
                transformed_data = TruncatedSVD(n_components=3).fit_transform(data)
            np.save(output_positions,transformed_data)
        except:
            pass
    del scaffolds_index, kmer_index


@click.pass_context
def estimate_phylogeny(ctx, work_dir, informative_diff_kmers_dict_pickle, informative_kmers_pickle, sparse_diff_kmer_matrix, kernel, n_neighbors_kmers, weights):
    # transpose matrix and estimate phylogeny
    kmer_graph_path = work_dir+'kmer_graph_outputs/'
    informative_diff_kmers = pickle.load(open(informative_diff_kmers_dict_pickle,'rb'))
    kmers = pickle.load(open(informative_kmers_pickle,'rb'))
    data = sps.load_npz(sparse_diff_kmer_matrix)
    try:
        os.makedirs(kmer_graph_path)
    except:
        pass
    kmer_reverse_lookup = defaultdict(list)
    for subgenome in informative_diff_kmers.keys():
        for kmer in informative_diff_kmers[subgenome]:
            kmer_reverse_lookup[kmer] = subgenome
    kmer_labels = np.vectorize(lambda x: kmer_reverse_lookup[x])(kmers)
    #kmer_class = {kmer_label: i for i,kmer_label in enumerate(set(kmer_labels))}
    #kmer_classes = np.vectorize(lambda x: kmer_class[x])(kmer_labels)
    pickle.dump(kmer_labels,open(work_dir+'diff_kmer_labels.p','wb'))
    kmer_matrix_transposed = data.transpose()
    kmer_pca = KernelPCA(n_components=3,kernel=kernel).fit_transform(StandardScaler(with_mean=False).fit_transform(kmer_matrix_transposed))
    np.save(work_dir+'transformed_diff_kmer_matrix_kmers.npy',kmer_pca)
    distance_matrix = pairwise_distances(kmer_pca,metric='euclidean') # change metric??
    distance_matrix = (distance_matrix + distance_matrix.T)/2
    #distance_matrix = pdist(kmer_pca)
    #print distance_matrix[distance_matrix != distance_matrix.T]
    np.save(work_dir+'kmer_pairwise_distances.npy',distance_matrix)
    nn_kmers = NearestNeighbors(n_neighbors=n_neighbors_kmers, metric = 'precomputed')
    nn_kmers.fit(kmer_pca)
    kmers_nn_graph = nn_kmers.kneighbors_graph(distance_matrix, mode = ('distance' if weights else 'connectivity'))
    sps.save_npz(work_dir+'kmers_nn_graph.npz',kmers_nn_graph)
    output_Dendrogram(work_dir+'kmer_pairwise_distances.npy', informative_kmers_pickle, kmer_graph_path)
    ctx.invoke(plotPositions, positions_npy=work_dir+'transformed_diff_kmer_matrix_kmers.npy', labels_pickle= informative_kmers_pickle, colors_pickle= work_dir+'diff_kmer_labels.p', output_fname=kmer_graph_path+'kmer_graph.html', graph_file= work_dir+'kmers_nn_graph.npz', layout= 'standard', iterations=25)
    # estimate phylogeny
    constructor = DistanceTreeConstructor()
    matrix = [row[:i+1] for i,row in enumerate(distance_matrix.tolist())]#[list(distance_matrix[i,0:i+1]) for i in range(distance_matrix.shape[0])]
    if len(set(kmers)) < len(kmers):
        new_kmer_names = []
        kmer_counter = defaultdict(lambda: 0)
        for kmer in kmers:
            kmer_counter[kmer]+=1
            if kmer_counter[kmer] > 1:
                new_kmer_names.append(kmer+'_'+str(kmer_counter[kmer]))
            else:
                new_kmer_names.append(kmer)
        kmers = new_kmer_names
    print kmers
    dm = _DistanceMatrix(names=kmers,matrix=matrix)
    tree = constructor.nj(dm)
    Phylo.write(tree,work_dir+'output_kmer_tree_by_scaffolds.nh','newick')


def TE_analysis(repeat_folder, TE_bed, output_subgenomes_bed, TE_sparse_matrix):
    import seaborn as sns
    # intersect TE's with subgenome labeled regions
    TE_bed = BedTool(TE_bed)
    subgenomes = BedTool(output_subgenomes_bed)
    subgenome_TE_bed = TE_bed.intersect(subgenomes, wa = True, wb = True)#.merge(c=7, o='distinct')
    ambiguous_TE_bed = TE_bed.intersect(subgenomes, v=True, wa = True)
    subgenome_bed_text = ''
    for line in str(subgenome_TE_bed).splitlines():
        if line:
            lineList = line.strip('\n').split()
            subgenome_bed_text += '\t'.join(['_'.join(lineList[0:3]), '0', str(int(lineList[2]) - int(lineList[1])), lineList[-1]]) + '\n'
    for line in str(ambiguous_TE_bed).splitlines():
        if line:
            lineList = line.strip('\n').split()
            subgenome_bed_text += '\t'.join(['_'.join(lineList[0:3]), '0', str(int(lineList[2]) - int(lineList[1])), 'ambiguous']) + '\n'
    subgenome_TE_bed = BedTool(subgenome_bed_text,from_string=True).sort().merge(c=4, o='distinct')
    TE_subgenomes = defaultdict(list)
    for line in str(subgenome_TE_bed).splitlines():
        if line:
            lineList = line.strip('\n').split()
            TE_subgenomes[lineList[0]] = (lineList[-1] if ',' not in lineList[-1] else 'ambiguous')
    TE_classes = pickle.load(open(repeat_folder + 'repeat_class_dictionary.p','rb'))

    TE_index = TE_classes.keys()
    data = sps.load_npz(TE_sparse_matrix)
    if 1:
        sum_kmer_counts = np.array(data.sum(axis=1))[:,0]
        print sum_kmer_counts
        plots= []
        for subclass in [0,1]:
            kmer_sums_TE_class = defaultdict(list)
            if subclass:
                for i in range(len(sum_kmer_counts)):
                    kmer_sums_TE_class[TE_classes[TE_index[i]]].append(sum_kmer_counts[i])
            else:
                for i in range(len(sum_kmer_counts)):
                    kmer_sums_TE_class[TE_classes[TE_index[i]].split('/')[0]].append(sum_kmer_counts[i])
            for TE_class in kmer_sums_TE_class:
                if type(kmer_sums_TE_class[TE_class]) != type(list):
                    kmer_sums_TE_class[TE_class] = sum(kmer_sums_TE_class[TE_class])
            plots.append(go.Bar(x=kmer_sums_TE_class.keys(),y=kmer_sums_TE_class.values(),name=('Subclasses' if subclass else 'Classes')))
            class_vs_subgenome = defaultdict(list)
            if subclass:
                 for TE in TE_classes:
                    class_vs_subgenome[TE_classes[TE]].append(TE_subgenomes[TE])
            else:
                for TE in TE_classes:
                    class_vs_subgenome[TE_classes[TE].split('/')[0]].append(TE_subgenomes[TE])
            for TE_class in class_vs_subgenome:
                if type(class_vs_subgenome[TE_class]) != type(list):
                    class_vs_subgenome[TE_class] = Counter(class_vs_subgenome[TE_class])
            df = pd.DataFrame(class_vs_subgenome)
            df.fillna(0, inplace=True)
            indep_test_results = chi2_contingency(df.as_matrix(), correction=True)
            df.to_csv((repeat_folder + 'TE_subclass_vs_subgenome.csv' if subclass else repeat_folder + 'TE_class_vs_subgenome.csv'))
            plt.figure(figsize=(13,10))
            sns_plot = sns.heatmap(df, annot=True)
            plt.title(r'TE %s v Intersected Subgenomes, p=%.2f, ${\chi}^2$=%.2f'%(('Subclasses' if subclass else 'Classes'),indep_test_results[1],indep_test_results[0]))
            plt.xlabel(('TE Subclasses' if subclass else 'TE Classes'))
            plt.ylabel('TE Subgenomes')
            plt.savefig(repeat_folder + ('TE_subclass_vs_subgenome.png' if subclass else 'TE_class_vs_subgenome.png'))
            traces = []
            for TE_class in class_vs_subgenome:
                 if type(class_vs_subgenome[TE_class]) != type(list):
                     traces.append(go.Bar(x=class_vs_subgenome[TE_class].keys(),y=class_vs_subgenome[TE_class].values(),name=TE_class))
            fig = go.Figure(data=traces,layout=go.Layout(barmode='group'))
            py.plot(fig,filename=repeat_folder+('TE_subclass_vs_subgenome.html' if subclass else 'TE_class_vs_subgenome.html'), auto_open=False)
        fig = go.Figure(data=plots,layout=go.Layout(barmode='group'))
        py.plot(fig, filename=repeat_folder+'Top_Differential_Kmer_Counts_By_Class.html', auto_open=False)

    subclasses = np.array(TE_classes.values())
    classes = np.vectorize(lambda x: x.split('/')[0])(subclasses)
    data = data.transpose()
    kmer_class_matrix = []
    for TE_class in set(classes):
        if TE_class != 'Unknown':
            kmer_class_matrix.append(np.array(data[:,classes == TE_class].sum(axis=1))[:,0]) # FIXME!!! and change if 0 back to normal
    print kmer_class_matrix

    classes = list(set(classes) - {'Unknown'})
    kmer_class_matrix = np.vstack(kmer_class_matrix).T
    print kmer_class_matrix
    kmer_classes = Counter(list(np.vectorize(lambda x: classes[x])(np.argmax(kmer_class_matrix,axis=1))))
    pickle.dump(kmer_classes,open(repeat_folder+'TE_kmer_classes.p','wb'))
    kmer_subclass_matrix = []
    for TE_class in set(subclasses):
        if TE_class != 'Unknown':
            kmer_subclass_matrix.append(np.array(data[:,subclasses == TE_class].sum(axis=1))[:,0])
    subclasses = list(set(subclasses) - {'Unknown'})
    kmer_subclass_matrix = np.vstack(kmer_subclass_matrix).T
    kmer_subclasses = Counter(list(np.vectorize(lambda x: subclasses[x])(np.argmax(kmer_subclass_matrix,axis=1))))
    pickle.dump(kmer_classes,open(repeat_folder+'TE_kmer_subclasses.p','wb'))
    plots = []
    for kmer_counts, label in [(kmer_classes,'Classes'),(kmer_subclasses,'Subclasses')]:
        plots.append(go.Bar(x = kmer_counts.keys(),y = kmer_counts.values(), name=label))
    fig = go.Figure(data=plots,layout=go.Layout(barmode='group'))
    py.plot(fig, filename=repeat_folder+'Differential_Kmers_By_Class.html', auto_open=False)
    # FIXME finish!!! ^^^^ can identify individual kmers that belong to each class

    #ctx.invoke(plotPositions,positions_npy=work_dir+'transformed_diff_kmer_matrix.npy', labels_pickle=work_dir+'scaffolds_diff_kmer_analysis.p', colors_pickle=work_dir+'jaccard_clustering_labels.p', output_fname=work_dir+'jaccard_clustering_results/jaccard_plot.html', graph_file=work_dir+'jaccard_nn_graph.npz', layout='positions', iterations=25)


@repeats.command()
@click.option('-w', '--work_dir', default='./diff_kmer_distances/', show_default=True, help='Work directory where computations for differential kmer analysis is done.', type=click.Path(exists=False))
@click.option('-go', '--original_genome', help='Full path to original polyploid fasta, prechunked.', type=click.Path(exists=False))
def avg_distance_between_diff_kmers(work_dir, original_genome):
    """Report the average distance between iterations of unique differential kmers over any scaffold."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    blast_path = work_dir + 'blast_files/'
    bed_path = work_dir + 'bed_alt_files/'
    output_path = work_dir+'outputs_distances/'
    if 1: #FIXME test
        for dir in [work_dir, blast_path, bed_path,output_path]:
            try:
                os.makedirs(dir)
            except:
                pass
    blast2bed3(work_dir, blast_path, bed_path, output_path, original_genome, 1, bedgraph=0, out_kmers = 1)
    subprocess.call('cat %s | bedtools sort > %s'%(' '.join([bed_path+file for file in os.listdir(bed_path) if file.endswith('.bed3')]),work_dir+'diff_kmer_locations.bed'),shell=True)
    fai_file = original_genome+'.fai'
    with open(fai_file,'r') as f:
        distance_dict = {line.split()[0]:defaultdict(list) for line in f}
    with open(work_dir+'diff_kmer_locations.bed','r') as f:
        for line in f:
            ll = line.split()
            distance_dict[ll[0]][ll[-1].strip('\n')].append(int(ll[1]))
    kmer_avg_distances = defaultdict(list)
    for chrom in distance_dict:
        for kmer in distance_dict[chrom].keys():
            if len(distance_dict[chrom][kmer]) >= 2:
                kmer_avg_distances[kmer].extend(np.diff(sorted(distance_dict[chrom][kmer])).tolist())
                del distance_dict[chrom][kmer]
            else:
                del distance_dict[chrom][kmer]
    for kmer in kmer_avg_distances.keys():
        kmer_avg_distances[kmer] = np.mean(kmer_avg_distances[kmer])
    np.save(output_path+'Average_Top_Kmer_Distance.npy',np.array(stats(kmer_avg_distances.values())))
    plt.figure()
    sns.distplot(kmer_avg_distances.values(),kde=False)
    plt.title(' Average Distance Between Particular Top Differential Kmer')
    plt.xlabel('Average Distance Between Particular Top Differential Kmer')
    plt.ylabel('Count')
    plt.savefig(output_path+'Average_Top_Kmer_Distance.png',dpi=300)


@repeats.command()
@click.option('-w', '--work_dir', default='./diff_kmer_analysis/', show_default=True, help='Work directory where computations for differential kmer analysis is done.', type=click.Path(exists=False))
@click.option('-npz', '--sparse_kmer_matrix', default='clusteringMatrix.npz', help='Original sparse kmer matrix produced from generate_Kmer_Matrix', show_default=True, type=click.Path(exists=False))
@click.option('-p', '--final_output_label_pickle', help='Pickle generated from bootstrap results from subgenomeExtraction. May need to run convert_subgenome_output_to_pickle.', type=click.Path(exists=False))
@click.option('-b', '--final_output_label_bed', help='Bed generated from bootstrap results from subgenomeExtraction. May need to run generate_out_bed.', type=click.Path(exists=False))
@click.option('-s', '--scaffolds_pickle', default='scaffolds.p', help='Scaffolds pickle produced from generate_Kmer_Matrix.', show_default=True, type=click.Path(exists=False))
@click.option('-kp', '--kmers_pickle', default='kmers.p', help='Kmers pickle produced from generate_Kmer_Matrix.', show_default=True, type=click.Path(exists=False))
@click.option('-min', '--min_number_diff_kmers', default=100, help='Minimum number of differential kmers from each subgenome.', show_default=True)
@click.option('-dd', '--diff_kmers_directory', help='Directory containing differential kmers from subgenomeExtraction bootstrap', show_default=True, type=click.Path(exists=False))
@click.option('-go', '--original_genome', help='Filename of original polyploid fasta, prechunked.', type=click.Path(exists=False))
@click.option('-fp', '--fasta_path', default='./fasta_files/', help='Path to original polyploid fasta, prechunked.', type=click.Path(exists=False))
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
@click.option('-sl', '--split_length', default=75000, help='Length of intervals in bedgraph files.', show_default=True)
@click.option('-n', '--n_chromosomes', default=10, help='Output unionbed plots of x largest chromosomes.', show_default=True)
@click.option('-s', '--n_subgenomes', default = 2, help='Number of subgenomes for Jaccard clustering.', show_default=True)
@click.option('-nns', '--n_neighbors_scaffolds', default=25, help='Number of nearest neighbors to use when generating nearest neighbor graph with scaffolds as datapoints.', show_default=True)
@click.option('-nnk', '--n_neighbors_kmers', default=15, help='Number of nearest neighbors to use when generating nearest neighbor graph with kmers as datapoints.', show_default=True)
@click.option('-k', '--kernel', default='cosine', help='Kernel for KPCA. Cosine is particularly effective.', type=click.Choice(['linear','poly','rbf','sigmoid','cosine']), show_default=True)
@click.option('-wnn', '--weights', default=0, help='Whether to weight k_neighbors graph for jaccard clustering and kmer graphs.', show_default=True)
@click.option('-r', '--repeat_masked_gff', default='xxx', help='Repeatmasked gff file, in gff or gff3 format. Used for intersecting kmer results with Transposable Elements to find TE classes for kmers and other analyses.', type=click.Path(exists=False))
@click.option('-ndb', '--no_diff_blast', 'no_run_diff', is_flag=True, help='Flag to not run earlier part of differential kmer analysis. Saves time if already done.' )
@click.option('-ns', '--no_sam2matrix', is_flag=True, help='Flag, if on, does not convert sam files into a clustering matrix.')
@click.option('-np', '--no_run_phylogeny', 'no_phylogeny', is_flag=True, help='Flag, if on, does not run phylogeny analysis')
@click.option('-nj', '--no_jaccard_clustering', 'no_jaccard', is_flag=True, help='Flag, if on, does not run jaccard clustering')
@click.pass_context
def diff_kmer_analysis(ctx, work_dir, sparse_kmer_matrix, final_output_label_pickle, final_output_label_bed, scaffolds_pickle, kmers_pickle, min_number_diff_kmers, diff_kmers_directory, original_genome, fasta_path, blast_mem, split_length, n_chromosomes, n_subgenomes, n_neighbors_scaffolds, n_neighbors_kmers, kernel, weights, repeat_masked_gff, no_run_diff, no_sam2matrix, no_phylogeny, no_jaccard):
    """Runs robust differential kmer analysis pipeline. Find highly informative differential kmers, subset of differential kmers found via polyCRACKER,
    generate a matrix of highly informative differential kmer counts versus scaffolds, generate a network graph linking the kmers,
    estimate phylogeny from these kmers, perform clustering using this matrix via a jaccard statistic, and look at the intersection between these kmers and identified repeats."""
    if work_dir.endswith('/') == 0:
        work_dir += '/'
    blast_path = work_dir + 'blast_files/'
    bed_path = work_dir + 'bed_files/'
    sort_path = work_dir + 'sorted_files/'
    output_images = work_dir + 'unionbed_plots/'
    repeat_path = work_dir + 'repeat_analysis/'
    if 1: #FIXME test
        for dir in [work_dir, blast_path, bed_path, sort_path, output_images, repeat_path]:
            try:
                os.makedirs(dir)
            except:
                pass
        # generate dictionaries containing differential kmers
        diff_kmers = defaultdict(list)
        for file in os.listdir(diff_kmers_directory):
            if file.endswith('.higher.kmers.fa'):
                with open(diff_kmers_directory+'/'+file,'r') as f:
                    diff_kmers[file[file.find('_')+1:file.find('.')]] = f.read().splitlines()[1::2]
        # generate labels for kbest selection
        output_labels = pickle.load(open(final_output_label_pickle,'rb'))
        sparse_kmer_matrix = sps.load_npz(sparse_kmer_matrix)
        labels_dict = {label:i for i, label in enumerate(set(output_labels) - {'ambiguous'})}
        sparse_kmer_matrix = sparse_kmer_matrix[output_labels != 'ambiguous']
        encoded_labels = np.vectorize(lambda x: labels_dict[x])(output_labels[output_labels != 'ambiguous'])
        # load all kmers
        kmers = pickle.load(open(kmers_pickle,'rb'))
        # run kbest to score the kmers
        kbest = SelectKBest(chi2,'all')
        kbest.fit(sparse_kmer_matrix,encoded_labels)
        kmers_ordered = np.array(kmers)[kbest.pvalues_.argsort()]
        # find informative differential kmers
        informative_diff_kmers = defaultdict(list)
        for subgenome in diff_kmers:
            if np.mean(np.vectorize(len)(diff_kmers[subgenome])) != np.mean(np.vectorize(len)(kmers_ordered)):
                print 'Not the same length of kmers between initial scaffolding and subgenome Extraction. The kmer sets must have matching lengths'
                quit()
            informative_diff_kmers[subgenome] = [kmer for kmer in kmers_ordered if kmer in diff_kmers[subgenome]][:min_number_diff_kmers]
        pickle.dump(informative_diff_kmers, open(work_dir+'informative_diff_kmers_dict.p','wb'))
        kmers = list(itertools.chain.from_iterable(informative_diff_kmers.values()))
        pickle.dump(kmers, open(work_dir+'informative_diff_kmers_list.p','wb'))

    repeat_bed = repeatGFF2Bed(repeat_masked_gff, repeat_path)

    if no_run_diff == 0:
        kmer_dir = write_informative_diff_kmers(informative_diff_kmers, work_dir)
        writeBlast(original_genome, blast_path, kmer_dir, fasta_path, 1, blast_mem)
        blast2bed3(work_dir, blast_path, bed_path, sort_path, original_genome, 1, bedgraph=0)
        ctx.invoke(generate_unionbed, bed3_directory=bed_path, original=1, split_length=split_length, fai_file=fasta_path+original_genome+'.fai', work_folder=work_dir)
        ctx.invoke(plot_unionbed,unionbed_file=work_dir+'subgenomes.union.bedgraph', number_chromosomes=n_chromosomes, out_folder=output_images)


    # intersect bed3 files with repeat density and generate matrix, feature labels are types of transposons, clustering matrix by known repeat regions
    # turn repeat density into list of scaffolds and corresponding list of features
    if no_sam2matrix == 0:
        sam2diffkmer_clusteringmatrix(work_dir,work_dir+'informative_diff_kmers_list.p',blast_path,kernel, repeat_bed)
    if no_phylogeny == 0:
        estimate_phylogeny(work_dir, work_dir+'informative_diff_kmers_dict.p', work_dir+'informative_diff_kmers_list.p', work_dir+'diff_kmer_sparse_matrix.npz', kernel, n_neighbors_kmers, weights)
    # convert to binary and output binary matrix for jaccard clustering
    if no_jaccard == 0:
        jaccard_clustering(work_dir, work_dir+'diff_kmer_sparse_matrix.npz', work_dir+'scaffolds_diff_kmer_analysis.p', n_subgenomes, n_neighbors_scaffolds, weights)
    if repeat_bed.endswith('.bed'):
        TE_analysis(repeat_path, repeat_bed, final_output_label_bed, repeat_path+'TE_sparse_matrix.npz')


@repeats.command(name='TE_Cluster_Analysis')
@click.option('-w', '--work_dir', help='Work directory where computations for TE clustering is done.', type=click.Path(exists=False))
@click.option('-rg', '--repeat_gff', help='Full path to repeat TE gff file, output from repeat masker pipeline.', type=click.Path(exists=False))
@click.option('-rp', '--repeat_pickle', help='Full path to repeat TE pickle file, containing TE names, which will be generated in this pipeline.', type=click.Path(exists=False))
@click.option('-sl', '--split_length', default=75000, help='Length of intervals in bedgraph files.', show_default=True)
@click.option('-n', '--n_chromosomes', default=10, help='Output unionbed plots of x largest chromosomes.', show_default=True)
@click.option('-s', '--n_clusters', default = 2, help='Number of clusters for clustering TEs.', show_default=True)
@click.option('-d', '--n_dimensions', default=3, help='Number of dimensions to reduce to.', show_default=True)
@click.option('-k', '--kernel', default='cosine', help='Kernel for KPCA. Cosine is particularly effective.', type=click.Choice(['linear','poly','rbf','sigmoid','cosine']), show_default=True)
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
@click.option('-nt', '--n_top_repeats', default=100, help='Number of top TEs in each subgenome to analyze.', show_default=True)
@click.option('-fai', '--fai_file', help='Full path to original, nonchunked fai file.', type=click.Path(exists=False))
@click.option('-dv', '--default_repeat_val', default=2, help='Default value for number of times TE appears in subgenome if value is 0.', show_default=True)
@click.option('-dt', '--diff_repeat_threshold', default=20, help='Threshold specifying that if one TE is differential in one subgenome versus others label that TE by that subgenome.', show_default=True)
@click.option('-bo','--original_subgenomes_bed', default = '', help='Full path to original bed file containing subgenome labelling. Will use this to label TEs by subgenome instead of clustering. Leave blank if deciding labels via clustering.', type=click.Path(exists=False))
@click.pass_context
def TE_Cluster_Analysis(ctx, work_dir, repeat_gff, repeat_pickle, split_length, n_chromosomes, n_clusters, n_dimensions, kernel, blast_mem, n_top_repeats, fai_file,default_repeat_val,diff_repeat_threshold, original_subgenomes_bed):
    """Build clustering matrix (repeat counts vs scaffolds) from repeats and conduct analyses on repeats instead of kmers."""
    if 1:
        if work_dir.endswith('/') == 0:
            work_dir += '/'
        bed_path = work_dir + 'bed3_files/'
        union_path = work_dir + 'unionbed_images/'
        for dir in [work_dir, bed_path, union_path]:
            try:
                os.makedirs(dir)
            except:
                pass
        #subprocess.call("awk '{print $1}' %s > %s/formatted_repeats.fa"%(repeat_fasta, work_dir), shell=True)
        # FIXME ADD ARGUMENTS BELOW
        repeatGFF2Bed(repeat_gff, work_dir, 1)
        with open(work_dir+'repeats.bed','r') as f, open(bed_path+'repeats.bed','w') as f2:
            for line in f:
                lineList = line.split()[0:2]
                f2.write('\t'.join([lineList[0],lineList[1],str(int(lineList[1])+1)])+'\n')
        ctx.invoke(generate_unionbed, bed3_directory=bed_path, original=1, split_length=split_length, fai_file=fai_file, work_folder=work_dir)
        ctx.invoke(plot_unionbed, unionbed_file=work_dir+'subgenomes.union.bedgraph', number_chromosomes=n_chromosomes, out_folder=union_path)
        sam2diffkmer_clusteringmatrix(work_dir,work_dir+'TE_motif_subclass.p',work_dir,kernel, work_dir+'repeats.bed', 1)
    scaffolds_pickle = work_dir+'scaffolds_TE_cluster_analysis.p'
    clustering_matrix = sps.load_npz(work_dir+'TEclusteringMatrix.npz')
    if original_subgenomes_bed:
        label_new_windows(work_dir, work_dir+'windows.bed' , original_subgenomes_bed)
        out_labels = pickle.load(open(work_dir+'new_labels.p','rb'))
        labels_dict = {label:i for i, label in enumerate(sorted(set(out_labels) - {'ambiguous'}))} #FIXME swapping names on accident
        #ctx.invoke(plotPositions,positions_npy=work_dir+'TE_pca.npy', labels_pickle=scaffolds_pickle, colors_pickle=work_dir+'new_labels.p', output_fname=work_dir+'TE_pca_clusters.html', graph_file='xxx', layout='standard', iterations=0)
    else:
        TE_pca = Pipeline([('ss',StandardScaler(with_mean=False)),('kpca',KernelPCA(n_components=n_dimensions, kernel = kernel))]).fit_transform(clustering_matrix)
        np.save(work_dir+'TE_pca.npy', TE_pca)
        ctx.invoke(plotPositions,positions_npy=work_dir+'TE_pca.npy', labels_pickle=scaffolds_pickle, colors_pickle='xxx', output_fname=work_dir+'TE_pca.html', graph_file='xxx', layout='standard', iterations=0)
        bgm = BayesianGaussianMixture(n_components=n_clusters)
        bgm.fit(TE_pca)
        labels = bgm.predict(TE_pca)
        pickle.dump(np.vectorize(lambda x: 'Subgenome_%d'%x)(labels),open(work_dir+'TE_subgenome_labels.p','wb'))
        ctx.invoke(plotPositions,positions_npy=work_dir+'TE_pca.npy', labels_pickle=scaffolds_pickle, colors_pickle=work_dir+'TE_subgenome_labels.p', output_fname=work_dir+'TE_pca_clusters.html', graph_file='xxx', layout='standard', iterations=0)

    # load all kmers
    TEs = np.array(pickle.load(open(work_dir+'TE_motif_subclass.p','rb')))
    # run kbest to score the kmers
    kbest = SelectKBest(chi2,'all')
    cm = clustering_matrix
    if original_subgenomes_bed:
        print clustering_matrix.shape
        clustering_matrix = clustering_matrix[out_labels != 'ambiguous']
        labels = np.vectorize(lambda x: labels_dict[x])(out_labels[out_labels != 'ambiguous'])
    kbest.fit(clustering_matrix,labels)
    TEs_ordered = TEs[kbest.pvalues_.argsort()]
    print clustering_matrix.shape
    clustering_matrix = clustering_matrix.transpose()
    subgenomes = ['Subgenome %d'%x for x in sorted(set(labels))]
    TE_subgenome_matrix = []
    for subgenome in set(labels):
        TE_subgenome_matrix.append(np.vectorize(lambda x: x if x else default_repeat_val)(np.array(clustering_matrix[:,labels == subgenome].sum(axis=1))[:,0]))
    TE_subgenome_matrix = np.vstack(TE_subgenome_matrix).T
    #print TE_subgenome_matrix
    #TE_max = np.argmax(TE_subgenome_matrix,axis=1)
    row_function = lambda row: subgenomes[np.argmax(row)] if (row[row!=np.max(row)] and np.all(np.vectorize(lambda x: np.float(np.max(row))/x >= diff_repeat_threshold)(row[row != np.max(row)]))) else 'ambiguous'
    #TE_subgenomes = np.vectorize(lambda x: subgenomes[x])(TE_max)
    #TE_subgenomes = np.apply_along_axis(row_function,1,TE_subgenome_matrix)
    TE_subgenomes = np.array([row_function(row) for row in TE_subgenome_matrix])
    print 'unambiguous=',len(TE_subgenomes[TE_subgenomes != 'ambiguous'])
    TE_subgenome_dict = {subgenome: [TE for TE in TEs_ordered if TE in list(TEs[TE_subgenomes == subgenome])][:n_top_repeats] for subgenome in subgenomes if list(TEs[TE_subgenomes == subgenome])}
    #print TE_subgenome_dict
    TEs_ordered = list(itertools.chain.from_iterable(TE_subgenome_dict.values()))
    #print TEs_ordered
    TE_bool = np.vectorize(lambda TE: TE in TEs_ordered)(TEs)
    TE_subgenome_matrix = np.hstack((TEs[TE_bool][:,None],TE_subgenome_matrix[TE_bool]))
    np.save(work_dir+'subgenome_differential_top_TEs.npy',TE_subgenome_matrix)
    pd.DataFrame(TE_subgenome_matrix,columns=(['TEs']+subgenomes)).to_csv('subgenome_differential_top_TEs.csv',index=False)

    pickle.dump(TE_subgenome_dict, open(work_dir+'informative_subgenome_TE_dict.p','wb'))
    pickle.dump(TEs_ordered, open(work_dir+'informative_subgenome_TE_list.p','wb'))
    if 1:#FIXME RIGHT HERE!! UNDER DEVELOPMENT
        print clustering_matrix.T.shape
        TE_diff_sparse_matrix = cm.T[np.vectorize(lambda TE: TE in TEs_ordered)(TEs)]
        print TE_diff_sparse_matrix.T.shape
        TE_pca = Pipeline([('ss',StandardScaler(with_mean=False)),('kpca',KernelPCA(n_components=n_dimensions, kernel = kernel))]).fit_transform(TE_diff_sparse_matrix.transpose())
        print TE_pca.shape
        np.save(work_dir+'top_TE_pca.npy', TE_pca)
        ctx.invoke(plotPositions,positions_npy=work_dir+'top_TE_pca.npy', labels_pickle=scaffolds_pickle, colors_pickle='xxx', output_fname=work_dir+'top_TE_pca.html', graph_file='xxx', layout='standard', iterations=0)
        bgm = BayesianGaussianMixture(n_components=n_clusters)
        bgm.fit(TE_pca)
        labels = bgm.predict(TE_pca)
        print labels.shape
        pickle.dump(np.vectorize(lambda x: 'Subgenome_%d'%x)(labels),open(work_dir+'top_TE_subgenome_labels.p','wb'))
        ctx.invoke(plotPositions,positions_npy=work_dir+'top_TE_pca.npy', labels_pickle=scaffolds_pickle, colors_pickle=work_dir+'top_TE_subgenome_labels.p', output_fname=work_dir+'top_TE_pca_clusters.html', graph_file='xxx', layout='standard', iterations=0)
        sps.save_npz(work_dir+'TE_diff_subgenomes_matrix.npz', TE_diff_sparse_matrix.transpose()) #FIXME RIGHT HERE!!

        ctx.invoke(estimate_phylogeny, work_dir=work_dir, informative_diff_kmers_dict_pickle=work_dir+'informative_subgenome_TE_dict.p', informative_kmers_pickle=work_dir+'informative_subgenome_TE_list.p', sparse_diff_kmer_matrix=work_dir+'TE_diff_subgenomes_matrix.npz', kernel=kernel, n_neighbors_kmers=15, weights=0) #FIXME RIGHT HERE!!

    try:
        TEs_counter = pickle.load(open(work_dir+'TE_motif_subclass_counter.p','rb'))
        top_TEs_counter = {}
        for TE in TEs_ordered:
            top_TEs_counter[TE] = TEs_counter[TE]
        TE_subclasses = np.vectorize(lambda x: x.split('#')[1])(TEs_ordered)
        TE_classes = np.vectorize(lambda x: x.split('/')[0])(TE_subclasses)
        TE_counts = np.vectorize(lambda x: top_TEs_counter[x])(TEs_ordered)
        n_TEs_subclass = Counter(TE_subclasses)
        n_TEs_class = Counter(TE_classes)
        TE_subclasses = {subclass:sum(TE_counts[TE_subclasses==subclass]) for subclass in set(TE_subclasses)}#Counter(TE_subclasses)
        TE_classes = {upper_class:sum(TE_counts[TE_classes==upper_class]) for upper_class in set(TE_classes)}#Counter(TE_classes)
        for class_name, class_dict in [('Class',TE_classes),('Subclasses',TE_subclasses)]:
            plots = []
            plots.append(go.Bar(x = class_dict.keys(),y = class_dict.values(), name=class_name))
            fig = go.Figure(data=plots,layout=go.Layout(barmode='group'))
            py.plot(fig, filename=work_dir+'TotalCounts_Top_TEs_By_%s.html'%(class_name), auto_open=False)
        for class_name, class_dict in [('Class',n_TEs_class),('Subclasses',n_TEs_subclass)]:
            plots = []
            plots.append(go.Bar(x = class_dict.keys(),y = class_dict.values(), name=class_name))
            fig = go.Figure(data=plots,layout=go.Layout(barmode='group'))
            py.plot(fig, filename=work_dir+'Top_TEs_By_%s.html'%(class_name), auto_open=False)
    except:
        print "Memory usage high. Will try to fix in later edition. Decrease number of top kmers."


@repeats.command(name='subgenome_extraction_via_repeats')
@click.pass_context
@click.option('-w', '--work_dir', default='./repeat_subgenome_extraction/', show_default=True, help='Work directory for computations of repeat supplied subgenome extraction.', type=click.Path(exists=False))
@click.option('-rc', '--repeat_cluster_analysis_dir', default='./TE_cluster_analysis/', show_default=True, help='Work directory where computations for initial TE clustering was done.', type=click.Path(exists=False))
@click.option('-cm', '--clustering_matrix', default='./TE_cluster_analysis/TEclusteringMatrix.npz', show_default=True, help='Repeats vs scaffolds matrix identified by TE_cluster_analysis.', type=click.Path(exists=False))
@click.option('-s','--scaffolds_pickle', default='./TE_cluster_analysis/scaffolds_TE_cluster_analysis.p', show_default=True, help='Pickle file containing the scaffold names from TE_cluster_analysis.',  type=click.Path(exists=False))
@click.option('-sl','--subgenome_labels_by_kmer_analysis', default = './diff_kmer_analysis/subgenomes.bed', show_default=True, help='Full path to original bed file containing subgenome labelling from polyCRACKER. Will use this to label TEs by subgenome instead of initial clustering. MANDATORY until future development.', type=click.Path(exists=False))
@click.option('-a','--all', is_flag=True, help='Grab all clusters.')
@click.option('-dv', '--default_repeat_val', default=1, help='Default value for number of times TE appears in subgenome if value is 0.', show_default=True)
@click.option('-dt', '--diff_repeat_threshold', default=15, help='Threshold specifying that if one TE is differential in one subgenome versus others label that TE by that subgenome.', show_default=True)
@click.option('-ds', '--differential_sum_ratio', default=7, help='Indicates that if the counts of all repeats in a particular region in any other subgenomes are 0, then the total count in a subgenome must be more than this much be binned as that subgenome.', show_default=True)
@click.option('-dc', '--default_total_count', default=2, help='Indicates that if any of the other subgenome kmer counts for all kmers in a region are greater than 0, then the counts on this subgenome divided by any of the other counts must surpass a value to be binned as that subgenome.', show_default=True)
@click.option('-k', '--kernel', default='cosine', help='Kernel for KPCA. Cosine is particularly effective.', type=click.Choice(['linear','poly','rbf','sigmoid','cosine']), show_default=True)
@click.option('-m', '--metric', default='cosine', help='Distance metric used to compute affinity matrix, used to find nearest neighbors graph for spectral clustering.', type=click.Choice(['cityblock','cosine','euclidean','l1','l2','manhattan','braycurtis','canberra','chebyshev','correlation','dice','hamming','jaccard','kulsinski','mahalanobis','matching','minkowski','rogerstanimoto','russellrao','seuclidean','sokalmichener','sokalsneath','sqeuclidean','yule']), show_default=True)
@click.option('-d', '--n_dimensions', default=3, help='Number of dimensions to reduce to.', show_default=True)
@click.option('-ns', '--n_subgenomes', default = 2, help='Number of subgenomes', show_default=True)
@click.option('-i', '--iterations', default = 3, help='Number of iterations for subgenome extraction bootstrapping.', show_default=True)
@click.option('-nn', '--n_neighbors', default=25, help='Number of nearest neighbors in generation of nearest neighbor graph.', show_default=True)
@click.option('-nt', '--n_top_repeats', default=100, help='Number of top TEs in each subgenome to analyze.', show_default=True)
@click.option('-rf', '--reference_fasta', default='', help='Full path to reference chunked fasta file containing scaffold names.', type=click.Path(exists=False))
def subgenome_extraction_via_repeats(ctx,work_dir, repeat_cluster_analysis_dir, clustering_matrix, scaffolds_pickle, subgenome_labels_by_kmer_analysis, all, default_repeat_val, diff_repeat_threshold, differential_sum_ratio, default_total_count, kernel, metric, n_dimensions, n_subgenomes, iterations, n_neighbors, n_top_repeats, reference_fasta):
    """Extends results of TE_cluster_analysis by performing subgenome extraction via matrix methods. Requires as input the found clustering matrix
    as identified via TE_cluster_analysis, and the subgenome labels from a polyCRACKER run output bed file."""
    import glob
    import matplotlib.pyplot as plt
    import seaborn as sns

    #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    work_dir += '/'
    repeat_cluster_analysis_dir += '/'
    try:
        os.makedirs(work_dir)
    except:
        pass
    clustering_matrix = sps.load_npz(clustering_matrix)
    scaffolds = pickle.load(open(scaffolds_pickle))
    scaffold_len = lambda scaffold: int(scaffold.split('_')[-1])-int(scaffold.split('_')[-2])
    weights = np.vectorize(scaffold_len)(scaffolds)
    kbest = SelectKBest(chi2,'all')
    cm = clustering_matrix
    TEs = np.arange(cm.shape[1])
    label_new_windows(work_dir, repeat_cluster_analysis_dir+'windows.bed' , subgenome_labels_by_kmer_analysis)
    out_labels = pickle.load(open(work_dir+'new_labels.p','rb'))
    labels_dict = {label:i for i, label in enumerate(sorted(set(out_labels) - {'ambiguous'}))}
    clustering_matrix = clustering_matrix[out_labels != 'ambiguous']
    labels = np.vectorize(lambda x: labels_dict[x])(out_labels[out_labels != 'ambiguous'])
    kbest.fit(clustering_matrix,labels)
    TEs_ordered = TEs[kbest.pvalues_.argsort()]
    clustering_matrix = clustering_matrix.transpose()
    subgenomes = ['Subgenome %d'%x for x in sorted(set(labels))]
    row_function = lambda row: subgenomes[np.argmax(row)] if (row[row!=np.max(row)] and np.all(np.vectorize(lambda x: np.float(np.max(row))/x >= diff_repeat_threshold)(row[row != np.max(row)]))) else 'ambiguous'

    if 0:
        TE_subgenome_matrix = []
        for subgenome in set(labels):
            TE_subgenome_matrix.append(np.vectorize(lambda x: x if x else default_repeat_val)(np.array(clustering_matrix[:,labels == subgenome].sum(axis=1))[:,0]))
        TE_subgenome_matrix = np.vstack(TE_subgenome_matrix).T
        TE_subgenomes = np.array([row_function(row) for row in TE_subgenome_matrix])
        TE_subgenome_dict = {subgenome: [TE for TE in TEs_ordered if TE in list(TEs[TE_subgenomes == subgenome])][:n_top_repeats] for subgenome in subgenomes if list(TEs[TE_subgenomes == subgenome])}
        TEs_ordered = list(itertools.chain.from_iterable(TE_subgenome_dict.values()))
        #print TEs_ordered
        TE_bool = np.vectorize(lambda TE: TE in TEs_ordered)(TEs)
        TE_subgenome_matrix = np.hstack((TEs[TE_bool][:,None],TE_subgenome_matrix[TE_bool]))
        TE_diff_sparse_matrix = cm.T[np.vectorize(lambda TE: TE in TEs_ordered)(TEs)]
        TE_diff_sparse_matrix_T = TE_diff_sparse_matrix.transpose()
        #OSVM = OneClassSVM()
        #OSVM.fit(TE_diff_sparse_matrix_T,sample_weight=weights)
        #inliers_outliers = OSVM.predict(TE_diff_sparse_matrix_T)
        #inliers = inliers_outliers == 1

        #encoded_labels = LabelEncoder().fit_transform(out_labels) StandardScaler(with_mean=False)
        TE_pca = Pipeline([('ss',StandardScaler(with_mean=False)),('kpca',KernelPCA(n_components=n_dimensions,kernel=kernel))]).fit_transform(TE_diff_sparse_matrix_T)#TfidfTransformer(),('tsne',TSNE(n_components=n_dimensions,n_jobs=8)).todense(),encoded_labels)# ('lda', LinearDiscriminantAnalysis(n_components=n_dimensions + 1)) lda based on number of classes fixme go back to kpca,('kpca',KernelPCA(n_components=n_dimensions, kernel = kernel))]).fit_transform(TE_diff_sparse_matrix.transpose()) # 25 ('tsne',TSNE(n_components=n_dimensions,n_jobs=8,metric=kernel if kernel == 'cosine' else 'euclidean',learning_rate=200,perplexity=50,angle=0.5)))#,('kpca',KernelPCA(n_components=n_dimensions, kernel = kernel))]).fit_transform(TE_diff_sparse_matrix.transpose()) # 25 ('tsne',TSNE(n_components=n_dimensions,n_jobs=8,metric=kernel if kernel == 'cosine' else 'euclidean',learning_rate=200,perplexity=50,angle=0.5))
        np.save(work_dir+'TE_pca.npy',TE_pca)
    TE_pca = np.load(work_dir+'TE_pca.npy')
    #TE_pca = TSNE(n_components=3,n_jobs=8).fit_transform(TE_pca)
    #np.save(work_dir+'TE_tsne.npy',TE_pca)
    ctx.invoke(plotPositions,positions_npy=work_dir+'TE_pca.npy', labels_pickle=scaffolds_pickle, colors_pickle=work_dir+'new_labels.p', output_fname=work_dir+'top_TE_pca_polycracker.html', graph_file='xxx', layout='standard', iterations=0)
    #pairwise = pairwise_distances(TE_pca,metric=metric)
    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm = 'auto', metric=metric)
    neigh.fit(TE_pca)
    nn_graph = neigh.kneighbors_graph(TE_pca, mode = 'connectivity')
    while sps.csgraph.connected_components(nn_graph)[0] > 1:
        print(n_neighbors, sps.csgraph.connected_components(nn_graph)[0])
        n_neighbors += 5
        neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm = 'auto', metric=metric)
        neigh.fit(TE_pca)
        nn_graph = neigh.kneighbors_graph(TE_pca, mode = 'connectivity')
    sps.save_npz(work_dir+'repeats_nn_graph.npz',nn_graph)
    fit_data = nn_graph

    """
    fit_data = nn_graph + sps.csgraph.minimum_spanning_tree(_fix_connectivity(TE_pca, nn_graph,  affinity = metric)[0].tocsr())
    fit_data += fit_data.T
    fit_data = (fit_data > 0).astype(np.float)"""
    #fit_data = neigh.kneighbors_graph(TE_pca, mode = 'connectivity')
    #mst = sps.csgraph.minimum_spanning_tree().tocsc()
    #fit_data += mst
    #fit_data += fit_data.T
    #fit_data = (fit_data > 0).astype(np.float)
    """
    G = nx.from_scipy_sparse_matrix(fit_data)
    plt.figure()
    nx.draw_spectral(G,node_color=encoded_labels,arrows=False,node_size=10)
    plt.savefig(work_dir+'spectral_layout.png',dpi=300)"""
    # fixme fix spectral clustering
    spec = SpectralClustering(n_clusters=n_subgenomes + 1 - int(all), affinity='precomputed', n_neighbors=n_neighbors)#'precomputed')
    spec.fit(fit_data)
    labels = spec.labels_

    #bgmm = BayesianGaussianMixture(n_components=n_subgenomes + 1 - int(all))
    #bgmm.fit(TE_pca)#[inliers])
    #labels = inliers_outliers
    #labels[inliers] = bgmm.predict(TE_pca[inliers])
    #labels = bgmm.predict(TE_pca)
    pickle.dump(labels,open(work_dir+'repeat_clusters.p','w'))
    ctx.invoke(plotPositions,positions_npy=work_dir+'TE_pca.npy', labels_pickle=scaffolds_pickle, colors_pickle=work_dir+'repeat_clusters.p', output_fname=work_dir+'top_TE_pca_clusters.html', graph_file='xxx', layout='standard', iterations=0)
    if not all:
        labels[labels == np.argmin(np.linalg.norm(map(lambda x: np.mean(x,axis=0),[TE_pca[labels == label] for label in sorted(set(labels))]),axis=1))] = -1
    try:
        os.makedirs(work_dir+'/preliminary/')
    except:
        pass
    for i in sorted(set(labels) - {-1}):#range(max(labels) + 1):
        output_scaffolds = scaffolds[labels == i]
        with open(work_dir+'/preliminary/subgenome_%d.txt'%i,'w') as f:
            f.write('\n'.join(output_scaffolds))
    cmT = cm.transpose()
    row_function2 = lambda row: np.argmax(row) if (row[row!=np.max(row)] and np.all(np.vectorize(lambda x: np.float(np.max(row))/x >= differential_sum_ratio)(row[row != np.max(row)]))) else -1
    n_repeats_each_extraction_run = []
    n_labelled_extraction_run = []
    for i in range(iterations):
        #try:
        TE_subgenome_matrix = []
        for subgenome in sorted(set(labels) - {-1}):#range(max(labels)+1):
            TE_subgenome_matrix.append(np.vectorize(lambda x: x if x else default_repeat_val)(np.array(cmT[:,labels == subgenome].sum(axis=1))[:,0]))
        TE_subgenome_matrix = np.vstack(TE_subgenome_matrix).T
        differential_TEs = np.array([row_function(row) for row in TE_subgenome_matrix])
        print('a')
        n_repeats_each_extraction_run.append([len(differential_TEs[differential_TEs == label]) for label in subgenomes])
        final_matrix = []
        for subgenome in subgenomes:
            final_matrix.append(np.vectorize(lambda x: x if x else default_total_count)(cm[:,differential_TEs == subgenome].sum(axis=1)))
        final_matrix = np.hstack(final_matrix)#.T
        print(final_matrix)
        labels = np.array([row_function2(row) for row in final_matrix])
        print('b')
        n_labelled_extraction_run.append([len(labels[labels==label]) for label in range(max(labels)+1)])
        #except:
        #    print 'iteration failure at %d, displaying next best results'%i
        #    break
    for run_analysis, outfile in zip([n_repeats_each_extraction_run,n_labelled_extraction_run],[work_dir+'/number_repeats_each_iteration.png',work_dir+'/number_labelled_scaffolds_each_iteration.png']):
        df = pd.DataFrame(run_analysis).reset_index(drop=False)
        print(df)
        df = pd.melt(df.rename(columns={'index':'Iteration'}),id_vars=['Iteration'],value_vars=range(max(labels) + 1)).rename(columns={'variable':'Subgenome','value':'Number_Hits/Labels'})
        print(run_analysis,df)
        plt.figure()
        sns.tsplot(data=df,condition='Subgenome',unit=None,interpolate=True,legend=True,value='Number_Hits/Labels',time='Iteration')
        plt.savefig(outfile,dpi=300)
    try:
        os.makedirs(work_dir+'/output/')
    except:
        pass
    scaffolds = np.array(pickle.load(open(scaffolds_pickle)))
    for i in sorted(set(labels) - {-1}):#range(max(labels) + 1):
        output_scaffolds = scaffolds[labels == i]
        with open(work_dir+'/output/subgenome_%d.txt'%i,'w') as f:
            f.write('\n'.join(output_scaffolds))

    pickle.dump(labels,open(work_dir+'repeat_final_partitions.p','w'))
    ctx.invoke(plotPositions,positions_npy=work_dir+'TE_pca.npy', labels_pickle=scaffolds_pickle, colors_pickle=work_dir+'repeat_final_partitions.p', output_fname=work_dir+'top_TE_pca_final.html', graph_file='xxx', layout='standard', iterations=0)
    if reference_fasta:
        ctx.invoke(txt2fasta,txt_files=','.join(glob.glob(work_dir+'/output/subgenome_*.txt')),reference_fasta=reference_fasta)


@repeats.command(name='differential_TE_histogram')
@click.option('-npy','--differential_repeat_subgenome_matrix','differential_TE_subgenome_matrix',default = './TE_cluster_analysis/subgenome_differential_top_TEs.npy',show_default=True, type=click.Path(exists=False))
def differential_TE_histogram(differential_TE_subgenome_matrix):
    """Compare the ratio of hits of certain highly informative repeats between different found genomes."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    diff_matrix = np.load(differential_TE_subgenome_matrix)[:,1:].astype(np.float)
    #print diff_matrix
    row_function = lambda row: np.array([np.argmax(row).astype(np.float),np.mean(np.vectorize(lambda x: np.float(np.max(row))/x)(row[row != np.max(row)]))])
    diff_matrix_differential = np.apply_along_axis(row_function,1,diff_matrix)
    #print diff_matrix_differential
    subgenomes = np.vectorize(lambda x: 'Subgenome %d'%int(x))(diff_matrix_differential[:,0])
    differentials = diff_matrix_differential[:,1]
    stats_diff_TEs = {}
    fig, axs = plt.subplots(1,len(set(subgenomes)),figsize=(7, 7), sharex=False, sharey=False)
    for i,subgenome in enumerate(set(subgenomes)):
        diff = differentials[subgenomes==subgenome]
        stats_diff_TEs[subgenome] = stats(diff)
        plt.axes(axs[i])
        sns.distplot(diff,label=subgenome,ax=axs[i],kde=False)
        plt.xlabel('Differential Score')
        plt.legend()
        if i == 0:
            plt.ylabel('Count')
    pd.DataFrame(stats_diff_TEs,index=['Mean','Standard Deviation', 'Min', 'Max']).to_csv('Differential_TEs_Diff_Score.csv')
    plt.title('How Differential are Differential TEs?')
    plt.savefig("Differential_TEs_Diff_Score.png",dpi=300)


def label_TE_bed(TE_bed, original_subgenomes_bed):
    TE_bed = BedTool(TE_bed).sort()
    subgenomes = BedTool(original_subgenomes_bed)
    subgenome_TE_bed = TE_bed.intersect(subgenomes, wa = True, wb = True).sort()
    ambiguous_TE_bed = TE_bed.intersect(subgenomes, v=True, wa = True).sort()
    subgenome_bed_text = ''
    for line in str(subgenome_TE_bed).splitlines():
        if line:
            lineList = line.strip('\n').split()
            subgenome_bed_text += '\t'.join(['_'.join(lineList[0:3]), '0', str(int(lineList[2]) - int(lineList[1])), lineList[-1]]) + '\n'
    for line in str(ambiguous_TE_bed).splitlines():
        if line:
            lineList = line.strip('\n').split()
            subgenome_bed_text += '\t'.join(['_'.join(lineList[0:3]), '0', str(int(lineList[2]) - int(lineList[1])), 'ambiguous']) + '\n'
    subgenome_TE_bed = BedTool(subgenome_bed_text,from_string=True).sort().merge(c=4, o='distinct')
    TE_subgenomes = defaultdict(list)
    for line in str(subgenome_TE_bed).splitlines():
        if line:
            lineList = line.strip('\n').split()
            TE_subgenomes[lineList[0]] = (lineList[-1] if ',' not in lineList[-1] else 'ambiguous')
    return TE_subgenomes


@repeats.command(name='repeat_subclass_analysis')
@click.option('-w', '--work_dir', default='./TE_subclass_analysis/', show_default=True, help='Work directory for computations of TE_subclass_analysis.', type=click.Path(exists=False))
@click.option('-go', '--original_fasta', help='Full path to original fasta file, nonchunked.', type=click.Path(exists=False))
@click.option('-rp', '--top_repeat_list_pickle', default='./TE_cluster_analysis/informative_subgenome_TE_list.p', help='Pickle containing list of top informative subgenome differential TEs.', show_default=True, type=click.Path(exists=False))
@click.option('-rd', '--top_repeat_dict_pickle', default='./TE_cluster_analysis/informative_subgenome_TE_dict.p', help='Pickle containing dict of top informative subgenome differential TEs by subgenome.', show_default=True, type=click.Path(exists=False))
@click.option('-r', '--all_family_subclass_pickle', default='./TE_cluster_analysis/TE_motif_subclass.p', help='Pickle containing set of all TEs family, subclasses.', show_default=True, type=click.Path(exists=False))
@click.option('-re', '--all_repeat_elements', default='./TE_cluster_analysis/repeat_elements.p', help='All repeat elements, to be converted to bed format.', show_default=True, type=click.Path(exists=False))
@click.option('-rs', '--class_subclass_pickle', default='./TE_cluster_analysis/repeat_class_subclass.p', help='All repeat classes and subclasses, to use to find indices to extract bed intervals.' ,show_default=True, type=click.Path(exists=False))
@click.option('-bo','--original_subgenomes_bed', default = '', help='Full path to original bed file containing subgenome labelling. Will use this to label TEs by subgenome.', type=click.Path(exists=False))
@click.option('-rf', '--repeat_fasta', default = '', help='Full path to repeat fasta file, output from repeat masker pipeline.', type=click.Path(exists=False))
@click.option('-bt','--bootstrap', is_flag=True, help='Bootstrap subclass trees generation.')
def repeat_subclass_analysis(work_dir, original_fasta, top_repeat_list_pickle, top_repeat_dict_pickle, all_family_subclass_pickle, all_repeat_elements, class_subclass_pickle, original_subgenomes_bed, repeat_fasta, bootstrap):
    """Input repeat_fasta and find phylogenies of TE subclasses within. In development: Better analysis may be https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2912890/."""
    #from Bio import AlignIO
    from ete3 import Tree, TreeStyle, TextFace #FIXME can't import treestyle, reset anaconda environment
    from xvfbwrapper import Xvfb
    if work_dir.endswith('/') == 0:
        work_dir += '/'
    try:
        os.makedirs(work_dir)
    except:
        pass
    colors = ["red","green","blue","orange","purple","pink","yellow"]
    if repeat_fasta:
        subprocess.call("awk '{print $1}' %s > %s && dedupe.sh overwrite=true in=%s out=%s ac=f && rm %s"%(repeat_fasta, work_dir+'temp.fa',work_dir+'temp.fa',work_dir+'repeats.fa',work_dir+'temp.fa'), shell=True)
    ### Subclass analysis
    all_TEs = np.array(pickle.load(open(all_family_subclass_pickle,'rb')))
    top_TEs = np.array(pickle.load(open(top_repeat_list_pickle,'rb')))
    all_TEs_subclasses = Counter(np.vectorize(lambda x: x.split('#')[1])(list(set(all_TEs))))
    subclasses = set(all_TEs_subclasses.keys())
    top_TEs_subclasses = Counter(np.vectorize(lambda x: x.split('#')[1])(list(set(top_TEs))))
    for subclass in subclasses:
        if subclass not in top_TEs_subclasses:
            del all_TEs_subclasses[subclass]
    all_TEs_subclasses = OrderedDict(sorted(all_TEs_subclasses.items()))
    top_TEs_subclasses = OrderedDict(sorted(top_TEs_subclasses.items()))
    plots = []
    plots.append(go.Bar(x = all_TEs_subclasses.keys(),y = all_TEs_subclasses.values(), name='All TEs'))
    plots.append(go.Bar(x = top_TEs_subclasses.keys(),y = top_TEs_subclasses.values(), name='Top TEs'))
    all_repeat_subclasses = np.array(all_TEs_subclasses.values())
    observed = np.array(top_TEs_subclasses.values())
    expected = np.array(all_TEs_subclasses.values())/float(sum(all_TEs_subclasses.values()))*sum(top_TEs_subclasses.values())
    plots.append(go.Bar(x = top_TEs_subclasses.keys(),y = expected, name='Expected TEs'))
    chisq = (observed-expected)**2/expected
    df = pd.DataFrame([all_repeat_subclasses,observed,expected,chisq],index=['all_repeat_subclasses','observed','expected','chisq'],columns=top_TEs_subclasses.keys())
    print df
    df.to_csv(work_dir+'/chi-sq-breakdown-subclasses.csv')
    top_chisq = np.argsort(chisq)[::-1]
    top_subclass = top_TEs_subclasses.keys()[top_chisq[0]]
    fig = go.Figure(data=plots,layout=go.Layout(barmode='group',title=r'All vs Top Differential TEs, by Subclass; Top Subclass: %s, chi^2=$%.2f'%(top_subclass,chisq[top_chisq[0]])))
    py.plot(fig, filename=work_dir+'Subclass_Division_Top_vs_All_TEs.html', auto_open=False)
    class_subclass = np.array(pickle.load(open(class_subclass_pickle,'rb')))
    all_repeats = np.array(pickle.load(open(all_repeat_elements,'rb')))
    repeats_bed = work_dir+'TE_elements.bed'
    with open(repeats_bed,'w') as f:
        for repeat in all_repeats:
            rlist = repeat.split('_')
            f.write('\t'.join(['_'.join(rlist[0:-2])]+rlist[-2:])+'\n')
    TE_labels = label_TE_bed(repeats_bed,original_subgenomes_bed)
    subgenomes_color = {subgenome: colors[i] for i, subgenome in enumerate(set(TE_labels.values()))}
    TE_colors = {TE: subgenomes_color[subgenome] for TE,subgenome in TE_labels.items()}
    top_repeat_dict = pickle.load(open(top_repeat_dict_pickle,'rb'))
    subgenomes_repeat_subclasses = {subgenome:defaultdict(lambda: 0) for subgenome in set(TE_labels.values())}
    for i,top_subclass in enumerate(np.array(top_TEs_subclasses.keys())[top_chisq]):
        top_subclass_repeats = all_repeats[class_subclass==top_subclass]
        top_subclass_fname = top_subclass.replace('/','-')
        if len(top_subclass_repeats) >= 2:
            Subgenome_type_counts = defaultdict(lambda: 0)
            for label in TE_labels:
                Subgenome_type_counts[TE_labels[label]] += 1
            plots = []
            plots.append(go.Bar(x = Subgenome_type_counts.keys(),y = Subgenome_type_counts.values(), name='Subgenomes'))
            fig = go.Figure(data=plots,layout=go.Layout(title='Subgenome Breakdown of All TEs Matching the %s most informative and interesting subclass # %d'%(top_subclass,i)))
            py.plot(fig, filename=work_dir+'Breakdown_All_subclass_%s_%d.html'%(top_subclass_fname,i), auto_open=False)
            #print TE_labels
            print list(set(TE_labels.values()))
            subclass_bed = work_dir+'subclass_%s.bed'%top_subclass_fname

            #### APPROACH 1
            if repeat_fasta:
                Subgenome_type_counts = defaultdict(lambda: 0)
                subclass_colors = defaultdict(list)
                for subgenome in top_repeat_dict:
                    subgenome_name = (subgenome[:1].lower() + ('_'.join(subgenome[1:].split())) if subgenome != 'ambiguous' else 'ambiguous')
                    for element in top_repeat_dict[subgenome]:
                        if element.split('#')[1] == top_subclass:
                            subclass_colors[element] = subgenomes_color[subgenome_name]
                            Subgenome_type_counts[subgenome_name] += 1
                for subgenome in subgenomes_repeat_subclasses:
                    try:
                        subgenomes_repeat_subclasses[subgenome][top_subclass] = Subgenome_type_counts[subgenome]
                    except:
                        subgenomes_repeat_subclasses[subgenome][top_subclass] = 0
                print subclass_colors
                plots = []
                plots.append(go.Bar(x = Subgenome_type_counts.keys(),y = Subgenome_type_counts.values(), name='Subgenomes'))
                fig = go.Figure(data=plots,layout=go.Layout(title='Subgenome Breakdown of Top TEs Matching the %s most informative and interesting subclass #%d'%(top_subclass,i)))
                py.plot(fig, filename=work_dir+'Breakdown_Top_subclass_%s_%d.html'%(top_subclass_fname,i), auto_open=False)
                top_subclass_elements = np.array(subclass_colors.keys())#top_TEs[np.vectorize(lambda x: x.split('#')[1] == top_subclass )(list(set(top_TEs)))]
                print len(top_TEs), top_subclass, len(top_subclass_elements)
                if len(top_subclass_elements) >= 2:# and top_subclass not in ['Unknown','Simple_repeat']:# and top_subclass != 'Simple_repeat':
                    subclass_fasta = work_dir+top_subclass_fname+'.fa'
                    print subclass_fasta
                    with open(subclass_fasta.replace('.fa','.intermediate.fa'),'w') as f2:
                        if top_subclass_fname != 'Simple_repeat':
                            with Fasta(work_dir+'repeats.fa') as f:
                                print top_subclass_elements
                                f2.write('\n'.join(['>%s\n%s'%(element,str(f[element][:])) for element in top_subclass_elements]))
                        else:
                            f2.write('\n'.join(['>%s\n%s'%(element,element[element.find('(')+1:element.find(')')]) if len(element.strip(')(')) != len(element) else '>%s\n%s'%(element,str(f[element][:])) for element in top_subclass_elements ]))
                    subprocess.call('reformat.sh overwrite=true in=%s out=%s fastawrap=60'%(subclass_fasta.replace('.fa','.intermediate.fa'),subclass_fasta),shell=True) # "export _JAVA_OPTIONS='-Xms5G -Xmx15G' && "+
                    #subprocess.call('samtools faidx %s %s > %s'%(work_dir+'repeats.fa', ' '.join(top_subclass_elements), subclass_fasta), shell=True)
                    vdisplay = Xvfb()
                    vdisplay.start()
                    subprocess.call('rm %stree_%s_output -r'%(work_dir,top_subclass_fname) if work_dir.strip(' ') != './' and '*' not in work_dir else 'echo change_work_dir',shell=True)
                    subprocess.call('ete3 build --cpu 2 -w %s -n %s -o %stree_%s_output --clearall'%('standard_phyml_bootstrap' if bootstrap else 'standard_fasttree',subclass_fasta,work_dir,top_subclass_fname),shell=True) # standard_phyml_bootstrap
                    tree = work_dir+'tree_%s_output/clustalo_default-none-none-%s/%s.final_tree.nw'%(top_subclass_fname,'phyml_default_bootstrap' if bootstrap else 'fasttree_full','%s.fa'%top_subclass_fname) #
                    print 'Still Running'
                    t = Tree(tree)
                    #t.link_to_alignment(alignment=subclass_fasta, alg_format="fasta")
                    for leaf in t:
                        leaf.img_style['size'] = 0
                        if leaf.is_leaf():
                            color=subclass_colors.get(leaf.name, None)
                            if color:
                                name_face = TextFace(leaf.name,fgcolor=color)
                                leaf.add_face(name_face,column=0,position='branch-right')
                    ts = TreeStyle()
                    #ts.mode = "c" # draw tree in circular mode
                    #ts.scale = 20
                    ts.show_leaf_name = False
                    t.render(work_dir+"subclass_%s_%d_tree.png"%(top_subclass_fname,i), dpi=300, w=1000, tree_style=ts)
                    print 'Debug Test'
                    vdisplay.stop()
    if repeat_fasta:
        plots = []
        for subgenome in subgenomes_repeat_subclasses:
            plots.append(go.Bar(x=subgenomes_repeat_subclasses[subgenome].keys(),y=subgenomes_repeat_subclasses[subgenome].values(),label=subgenome))
        fig = go.Figure(data=plots,layout=go.Layout(title='Subgenome/Subclass Breakdown of Top Repeats',barmode='stack'))
        py.plot(fig, filename=work_dir+'Breakdown_Subgenome_Subclass_Top.html', auto_open=False)
    else:
        print 'Breakdown No Print'
    #### APPROACH 2
    # find most commonly occurring top_subclass
    if 0:
        with open(subclass_bed,'w') as f:
            for repeat in top_subclass_repeats:
                rlist = repeat.split('_')
                f.write('\t'.join(['_'.join(rlist[0:-2])]+rlist[-2:] + [repeat])+'\n')
        subclass_fasta = subclass_bed.replace('.bed','.fasta')
        subprocess.call('bedtools getfasta -name -fi %s -bed %s -fo %s'%(original_fasta,subclass_bed,subclass_fasta),shell=True)
        subprocess.call('ete3 build --cpu 4 -w standard_fasttree -n %s  -o %stree_output --clearall'%(subclass_fasta,work_dir),shell=True)
        tree = work_dir+'tree_output/clustalo_default-none-none-fasttree_full/%s.final_tree.nw'%('subclass_%s.fasta'%top_subclass)
        t = Tree(tree)
        for leaf in t:
            leaf.add_features(color=TE_colors.get(leaf.name, "none"))
        ts = TreeStyle()
        ts.mode = "c" # draw tree in circular mode
        ts.scale = 20
        t.render("subclass_%s_tree.png"%top_subclass, w=183, units="mm", tree_style=ts)
        #subprocess.call('fftns %s > %s'%(work_dir+family+'.fa',work_dir+family+'_aligned.fasta'),shell=True)  mafft_ginsi-none-none-fasttree_default
        #subprocess.call('fasttree -nt -gtr < %s > %s'%(work_dir+family+'_aligned.phy',work_dir+family+'_tree.txt'), shell=True)
    ### FAMILY ANALYSIS, SKIP FOR NOW
    if 0:
        # format fasta file toupper()
        repeat_fasta = 'NULL, FIXME'
        subprocess.call("awk '{print $1}' %s > %s && dedupe.sh overwrite=true in=%s out=%s ac=f && rm %s"%(repeat_fasta, work_dir+'temp.fa',work_dir+'temp.fa',work_dir+'repeats.fa',work_dir+'temp.fa'), shell=True)

        # grab TE names
        TEs = Fasta(work_dir+'repeats.fa')
        #TE_names = np.array(TEs.keys())
        TE_names = np.array(pickle.load(open(top_repeat_list_pickle,'rb'))) #FIXME above is important, or do this by subgenome
        TE_families = np.vectorize(lambda x: x.split('_')[0] if x.endswith('rich') == 0 else x.split('#')[0])(TE_names)
        TE_families_set = set(TE_families)
        for family in TE_families_set:
            TEs_from_family = TE_names[TE_families==family]
            if len(TEs_from_family) >= 2:
                subprocess.call('samtools faidx %s %s > %s'%(work_dir+'repeats.fa', ' '.join(TEs_from_family), work_dir+family+'.fa'), shell=True)
                subprocess.call('fftns %s > %s'%(work_dir+family+'.fa',work_dir+family+'_aligned.fasta'),shell=True)
                subprocess.call('fasttree -nt -gtr < %s > %s'%(work_dir+family+'_aligned.phy',work_dir+family+'_tree.txt'), shell=True)
                #aln = AlignIO.read(open(work_dir+family+'_aligned.fasta','r'),'fasta')
                #AlignIO.write(aln,open(work_dir+family+'_aligned.phy','w'),'phylip-relaxed')
                #subprocess.call('raxmlHPC -s %s -n %s -m %s -f a -x 123 -N autoMRE -p 456 -q %s'%(work_dir+family+'_aligned.phy',work_dir+family+'_tree.txt','GTRCAT',work_dir+'bootstrap.'+family+'.txt'),shell=True)
        """MAYBE TRY mafft-qinsi"""


@repeats.command(name='send_repeats')
@click.option('-rf','--subclass_repeat_fasta',default='./TE_subclass_analysis/Unknown.fa',show_default=True,help='Full path to subclass repeat fasta.', type=click.Path(exists=False))
def send_repeats(subclass_repeat_fasta):
    """Use bbSketch to send fasta file containing repeats to check it against different databases for sequence similarity. Uses minhash to check relative abundance of species.
    Where do these repeats come from and what is the parasitic DNA source."""
    subprocess.call('sendsketch.sh in=%s nt overwrite=True'%(subclass_repeat_fasta),shell=True)#,subclass_repeat_fasta[:subclass_repeat_fasta.rfind('/')]),shell=True) # refseq mode=single
    subprocess.call('sendsketch.sh in=%s refseq overwrite=True'%(subclass_repeat_fasta),shell=True)#,subclass_repeat_fasta[:subclass_repeat_fasta.rfind('/')]),shell=True) # refseq mode=single


@repeats.command(name='run_iqtree')
@click.option('-f','--fasta_in',default='./TE_subclass_analysis/Unknown.fa',show_default=True,help='Full path to subclass repeat fasta. To be aligned via muscle.', type=click.Path(exists=False))
@click.option('-m','--model',default='MF',show_default=True,help='Model selection for iqtree. See http://www.iqtree.org/doc/Substitution-Models.')
@click.option('-nt','--n_threads',default='AUTO',show_default=True,help='Number of threads for parallel computation.')
@click.option('-b','--bootstrap',default=25,show_default=True,help='Bootstrap support.')
def run_iqtree(fasta_in,model,n_threads,bootstrap):
    """Perform multiple sequence alignment on multi-fasta and run iqtree to find phylogeny between each sequence."""
    import sys
    iqtree_line = next(path for path in sys.path if 'conda/' in path and '/lib/' in path).split('/lib/')[0]+'/bin/ete3_apps/bin/iqtree'
    muscle_line = next(path for path in sys.path if 'conda/' in path and '/lib/' in path).split('/lib/')[0]+'/bin/ete3_apps/bin/muscle'

    subprocess.call(muscle_line + ' -in %s -out %s'%(fasta_in,fasta_in.replace('.fasta','.muscle.fasta').replace('.fa','.muscle.fa')),shell=True)
    subprocess.call('rm %s.ckp.gz'%fasta_in.replace('.fasta','.muscle.fasta').replace('.fa','.muscle.fa'),shell=True)
    subprocess.call(iqtree_line + ' -s %s -m %s -nt %s %s'%(fasta_in.replace('.fasta','.muscle.fasta').replace('.fa','.muscle.fa'),model,n_threads,'-b %d'%bootstrap if bootstrap > 1 and model != 'MF' else ''), shell=True)


@repeats.command(name='find_denovo_repeats')
@click.option('-sp', '--input_species', help='Input species for naming of repeat database.', type=click.Path(exists=False))
@click.option('-fi', '--input_fasta', help='Full path to original fasta file, nonchunked.', type=click.Path(exists=False))
@click.option('-od', '--out_dir', default='./denovo_repeats/', show_default=True, help='Work directory for computations of denovo repeats.', type=click.Path(exists=False))
@click.option('-rd','--recover_dir', default='',show_default=True,help='Recover directory when using RepeatModeler',type=click.Path(exists=False))
def find_denovo_repeats(input_species,input_fasta, out_dir,recover_dir):
    """Wrapper for repeat modeler. Runs repeatmodeler and repeat masker to ID repeats."""
    print 'In development'
    if out_dir.endswith('/') == 0:
        out_dir += '/'
    try:
        os.makedirs(out_dir)
    except:
        pass
    recover_dir = (os.path.abspath(recover_dir) if recover_dir else '')
    if recover_dir:
        try:
            os.makedirs(recover_dir)
        except:
            pass
    input_fasta = os.path.abspath(input_fasta)
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    subprocess.call('BuildDatabase -name %s %s'%(input_species,input_fasta),shell=True)
    print 'nohup RepeatModeler %s -pa %d -database %s >& db.out'%('-recoverDir '+recover_dir if recover_dir else '',n_cpus-1,input_species)
    subprocess.call('nohup RepeatModeler %s -pa %d -database %s >& db.out'%('-recoverDir '+recover_dir if recover_dir else '',n_cpus-1,input_species),shell=True)
    print 'RepeatMasker -gff -dir %s -pa %d -lib %s-families.fa %s'%(out_dir,n_cpus-1,input_species,input_fasta)
    subprocess.call('RepeatMasker -gff -dir %s -pa %d -lib %s-families.fa %s'%(out_dir,n_cpus-1,input_species,input_fasta),shell=True)
    print 'Masker Done'
    # RUN TO THIS AND CONVERT TO GFF/BED
    # ADD GFF HERE
    # AND CHECK AGAINST NCBI
    #subprocess.call('sendsketch.sh in=%s nt mode=sequence out=%s/sketches.out',shell=True)
    # FIXME ADD INFO TO ABOVE


@repeats.command()
@click.pass_context
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory where computations for merging of clusters is done.', type=click.Path(exists=False))
@click.option('-rf', '--repeat_fasta', default = '', help='Full path to repeat fasta file, output from repeat masker pipeline.', type=click.Path(exists=False))
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
@click.option('-l', '--kmer_length', default=23, help='Length of kmers to find.', show_default=True)
@click.option('-r', '--rules_kmers', default = './kmer_master_matrix_rules.csv', show_default=True,help='Kmer count matrix with appended rule labels.', type=click.Path(exists=False))
@click.option('-gff', '--repeat_gff', default='', show_default=True, help='If repeat gff is submitted here, repeat fasta must be original genome, and non-consensus repeat sequences will be considered.', type=click.Path(exists=False))
@click.option('-motif', '--grab_motif', is_flag=True, help='Grab unique consensus repeat identifiers.')
@click.option('-dash', '--output_dash', is_flag=True, help='Output csv file for dash.')
def categorize_repeats(ctx,work_dir,repeat_fasta,blast_mem,kmer_length, rules_kmers, repeat_gff, grab_motif, output_dash):
    """Take outputs from denovo repeat finding/modeling and intersect kmers of particular conservation patterns with these repeats,
    either all iterations of a consensus repeat or consensus repeats themselves, and crosstabulate rule distribution with repeats/subclass."""
    from sklearn.preprocessing import LabelEncoder
    from pyfaidx import Fasta
    from sklearn.feature_extraction import DictVectorizer
    #from scipy.stats.contingency import expected_freq
    from sklearn.preprocessing import Normalizer
    import seaborn as sns

    try:
        os.makedirs(work_dir)
    except:
        pass

    def count_append(arr):
        d = defaultdict(lambda: 0)
        final_arr = []
        for element in arr.tolist():
            d[element] += 1
            final_arr.append(str(d[element])+'#'+element)
        return np.array(final_arr)

    def kmer_dict_gen(list_tuples):
        d = defaultdict(list)
        for key, val in list_tuples:
            d[key].append(val)
        return d

    def combine_dicts(d1,d2):
        return dict((k, d1.get(k,[]) + d2.get(k,[])) for k in set(d1).union(d2))

    # step one: classify repeats by kmer distribution to ID additional repeats belonging to subclass
    # step two: label these repeats by their kmers and find distribution of kmer rules for each subclass
    # FIXME reason why
    if repeat_gff:
        # extract sequences from original genome fasta and make new fasta file, call that one the repeat fasta
        subclass_motif = 'ID' if grab_motif else 'Subclass'
        df = pd.read_table(repeat_gff,sep='\t',skiprows=3,header=None,names=['Chr','xi','xf',subclass_motif],dtype={'Chr':str,'xi':float,'xf':str,subclass_motif:str},usecols=[0,3,4,8])#,columns=['Chr']+['x']*2+['xi','xf']+['x']*7+['Subclass'])
        #print df
        #df['xi':'xf'] = df['xi':'xf'].astype(int)
        df = df[np.isfinite(df['xi'])]
        df['xi'] = np.vectorize(lambda x: int(x)-1)(df['xi'])
        if grab_motif:
            df['ID'] = count_append(np.vectorize(lambda x: ':'.join(np.array(x.split())[[1,-1]]).replace('"','').replace('Motif:',''))(df['ID']))
        else:
            df['Subclass'] = count_append(np.vectorize(lambda x: x.split()[-1].replace('"',''))(df['Subclass']))
        #print df
        #df = df[['Chr','xi','xf','Subclass']]
        df.to_csv(work_dir+'repeat_extract.bed',sep='\t',index=False, header = False)
        subprocess.call('samtools faidx %s && bedtools getfasta -fi %s -fo %s -bed %s -name'%(repeat_fasta,repeat_fasta,work_dir+'extracted_repeats.fa',work_dir+'repeat_extract.bed'),shell=True)
        repeat_fasta = work_dir+'extracted_repeats.fa'
        repeat_names = df[subclass_motif].as_matrix()

    fasta = Fasta(repeat_fasta)
    if not repeat_gff:
        repeat_names = np.array(fasta.keys())
    blast_memStr = "export _JAVA_OPTIONS='-Xms5G -Xmx%sG'"%(blast_mem)
    if repeat_gff:
        df = pd.read_csv(rules_kmers,index_col=0)
        kmers = list(df.index)
        rule_mapping = zip(kmers,np.vectorize(lambda x: x.replace(' ','_'))(df['Rules']))
        with open(work_dir+repeat_fasta[repeat_fasta.rfind('/')+1:]+'.kcount.fa','w') as f:
            f.write('\n'.join(['>%s\n%s'%(rule+'@'+kmer,kmer) for kmer,rule in rule_mapping]))
    else:
        subprocess.call(blast_memStr + '&& kmercountexact.sh overwrite=true fastadump=f in=%s out=%s.kcount k=%d -Xmx100g'%(repeat_fasta,repeat_fasta,kmer_length),shell=True)
        subprocess.call("awk '{print $1}' %s > %s && dedupe.sh overwrite=true in=%s out=%s ac=f && rm %s"%(repeat_fasta, work_dir+'temp.fa',work_dir+'temp.fa',work_dir+'repeats.fa',work_dir+'temp.fa'), shell=True)

        #FIXME WIP below, also add splitting cluster ability
        ctx.invoke(kmer2Fasta,kmercount_path=work_dir, kmer_low_count=0, high_bool=0, kmer_high_count=100000000, sampling_sensitivity=1)
    ctx.invoke(blast_kmers,blast_mem=blast_mem,reference_genome=repeat_fasta,kmer_fasta=work_dir+repeat_fasta[repeat_fasta.rfind('/')+1:]+'.kcount.fa', output_file=work_dir+'repeat_blasted.sam')
    if output_dash:
        subprocess.call("awk -v OFS='\\t' '{split($6, a, "+'"="'+"); print $3, $4, $4 + a[1], $1 }' %s > %s"%(work_dir+'repeat_blasted.sam',work_dir+'repeat_blasted_dash.bed'),shell=True)
        repeat_lengths = dict(zip(os.popen("awk '{print $1}' %s"%repeat_fasta+'.fai').read().splitlines(),map(float,os.popen("awk '{print $2}' %s"%repeat_fasta+'.fai').read().splitlines())))
        df = pd.read_table(work_dir+'repeat_blasted_dash.bed',header=None,names=['Repeat','xi','xf','rule_kmer'],dtype=dict(Repeat=str,xi=float,xf=float,rule_kmer=str))
        df['length'] = np.vectorize(lambda x: repeat_lengths[x])(df['Repeat'])
        df['xm'] = (df['xi'] + df['xf']) / 2.
        df['rule'] = np.vectorize(lambda x: x.split('@')[0])(df['rule_kmer'])
        df['kmer'] = np.vectorize(lambda x: x.split('@')[1])(df['rule_kmer'])
        df['iteration'] = np.vectorize(lambda x: int(x.split('#')[0]))(df['Repeat'])
        df['Repeat'] = np.vectorize(lambda x: x.split('#')[1])(df['Repeat'])
        df['Subclass'] = np.vectorize(lambda x: x.split(':')[-1])(df['Repeat'])
        df['xm_length'] = df['xm']/df['length']
        df = df.drop(['xi','xf','rule_kmer'],axis=1).reindex(columns=['Subclass','Repeat','iteration','xm','length','xm_length','rule','kmer'])
        df.to_csv(work_dir+'dash_repeat_kmer_content.csv')
    else:
        ctx.invoke(blast2bed,blast_file=work_dir+'repeat_blasted.sam', bb=1 , low_memory=0, output_bed_file=work_dir+'repeat_blasted.bed', output_merged_bed_file=work_dir+'repeat_blasted_merged.bed', external_call=True)
        if not repeat_gff:
            kmers = np.array(os.popen("awk '{ print $1 }' %s"%(work_dir+repeat_fasta[repeat_fasta.rfind('/')+1:]+'.kcount')).read().splitlines())
            df = pd.read_csv(rules_kmers,index_col=0)
            excluded_kmers = set(kmers) - set(list(df.index))
            rule_mapping = dict(zip(list(df.index),df['Rules'])+zip(excluded_kmers,['rule -1']*len(excluded_kmers)))
            del df
        print repeat_names
        repeat_mapping = LabelEncoder()
        repeat_mapping.fit(repeat_names)
        if repeat_gff:
            subclasses = np.vectorize(lambda x: x.split('#')[1])(np.array(os.popen("awk '{ print $1 }' %s"%work_dir+'repeat_blasted_merged.bed').read().splitlines()))
            v = DictVectorizer(sparse=False,dtype=np.int)
            rule_encoding_info = [map(lambda x: tuple(x.split('@')),line.split(',')) for line in os.popen("awk '{ print $4 }' %s"%work_dir+'repeat_blasted_merged.bed').read().splitlines()]
            X = v.fit_transform([Counter(map(lambda x: x[0],line_info)) for line_info in rule_encoding_info])
            print X
            rule_set = v.get_feature_names()
            print rule_set
            subclass_set = set(subclasses)
            print subclass_set
            kmer_dicts = defaultdict(list)
            for i, list_tuples in enumerate(rule_encoding_info):
                kmer_dicts[subclasses[i]].append(kmer_dict_gen(list_tuples))
            for subclass in kmer_dicts:
                kmer_dicts[subclass] = { k:'\n'.join(['%s:%d'%(kmer,kcount) for kmer,kcount in Counter(v).items()]) for k,v in reduce(lambda x,y: combine_dicts(x,y),kmer_dicts[subclass]).items() }
            df_final2 = pd.DataFrame(kmer_dicts).transpose()
            df_final = pd.DataFrame(np.vstack(tuple([np.sum(X[subclasses == subclass,:],axis=0) for subclass in subclass_set])),index=list(subclass_set),columns=rule_set)
            df_final2 = df_final2.reindex(index=list(df_final.index.values),columns=list(df_final))
            subclass_counter = Counter(subclasses)
            df_final2['Element_Count'] = np.vectorize(lambda x: subclass_counter[x])(list(df_final2.index.values))
            df_final2 = df_final2.drop(['rule_-1'],axis=1)
            df_final2.to_csv(work_dir+'kmers_in_repeat_rule.csv')
        else:
            repeats_idx_bed = repeat_mapping.transform(np.array(os.popen("awk '{ print $1 }' %s"%work_dir+'repeat_blasted_merged.bed').read().splitlines()))
            subclasses = np.vectorize(lambda x: x.split('#')[1])(repeat_names)
            kmer_mapping = LabelEncoder()
            kmer_mapping.fit(kmers)
            data = sps.dok_matrix((len(repeat_names),len(kmers)),dtype=np.int)
            with open(work_dir+'repeat_blasted_merged.bed','r') as f:
                for i, line in enumerate(f):
                    ll = line.strip('\n').split()
                    d = Counter(ll[-1].split(','))
                    data[repeats_idx_bed[i],kmer_mapping.transform(d.keys())] = np.array(d.values())
            data = data.tocsr()
            rules = np.vectorize(lambda x: rule_mapping[x])(kmer_mapping.inverse_transform(range(len(kmers))))
            sparse_df = pd.SparseDataFrame(data,index=subclasses,columns=rules,default_fill_value=0)
            #print sparse_df
            df_final = sparse_df.groupby(sparse_df.columns, axis=1).sum().groupby(sparse_df.index).sum().to_dense().fillna(0)
        df_final = df_final.drop(['rule_-1'],axis=1)
        df_final.to_csv(work_dir+('repeat_rule_cross_tab.csv' if grab_motif and repeat_gff else 'subclass_rule_cross_tab.csv'))
        observed = df_final.as_matrix()
        pickle.dump(np.array(list(df_final)),open(work_dir+'rules.p','wb'))
        pickle.dump(np.array(list(df_final.index.values)),open(work_dir+'subclass.p','wb'))
        if not grab_motif:
            chi_sq, p, dof, expected = chi2_contingency(observed)
            plt.figure()
            sns.heatmap(observed/observed.sum(axis=1)[:,np.newaxis])
            plt.title('Chi-sq = %f, p=%f'%(chi_sq,p))
            plt.savefig(work_dir+'subclass_rule_cross_tab.png',dpi=300)
            plt.figure()
            ch2 = (observed-expected)**2/expected
            heat_plt = sns.heatmap(ch2)
            heat_plt.set(xticklabels=list(df_final),yticklabels=list(df_final.index.values))
            plt.title('Chi-sq = %f, p=%f'%(chi_sq,p))
            plt.savefig(work_dir+'subclass_rule_cross_tab_chi2.png',dpi=300)
            t_data = Pipeline([('norm',Normalizer()),('kpca',KernelPCA(n_components=3))]).fit_transform(observed)
            np.save(work_dir+'subclass_rules.npy',t_data)
            t_data = Pipeline([('norm',Normalizer()),('kpca',KernelPCA(n_components=3))]).fit_transform(observed.T)
            np.save(work_dir+'rules_subclass.npy',t_data)
            ctx.invoke(plotPositions,positions_npy=work_dir+'subclass_rules.npy',labels_pickle=work_dir+'subclass.p',colors_pickle='',output_fname=work_dir+'subclass_rules.html')
            ctx.invoke(plotPositions,positions_npy=work_dir+'rules_subclass.npy',labels_pickle=work_dir+'rules.p',colors_pickle='',output_fname=work_dir+'rules_subclass.html')
            #chi_sq, p, dof, expected = chi2_contingency(df_final.as_matrix()[:,1:])
            df_final.iloc[:,:] = expected#expected_freq(df_final.as_matrix()[:,1:])
            df_final.to_csv(work_dir+'subclass_rule_cross_tab_exp.csv')
            df_final.iloc[:,:] = ch2
            df_final.to_csv(work_dir+'subclass_rule_cross_tab_chi2.csv')
        elif grab_motif and repeat_gff:
            t_data = Pipeline([('norm',Normalizer()),('kpca',KernelPCA(n_components=3))]).fit_transform(observed)
            np.save(work_dir+'repeat_rules.npy',t_data)
            t_data = Pipeline([('norm',Normalizer()),('kpca',KernelPCA(n_components=3))]).fit_transform(observed.T)
            np.save(work_dir+'rules_repeat.npy',t_data)
            ctx.invoke(plotPositions,positions_npy=work_dir+'repeat_rules.npy',labels_pickle=work_dir+'subclass.p',colors_pickle='',output_fname=work_dir+'repeat_rules.html')
            ctx.invoke(plotPositions,positions_npy=work_dir+'rules_repeat.npy',labels_pickle=work_dir+'rules.p',colors_pickle='',output_fname=work_dir+'rules_repeat.html')
        # FIXME #1 maybe first establish voting system for consensus repeats, assign rules to consensus repeats
        # FIXME look at all consensus repeat types, find phylogeny, see if there are clade specific patterns in consensus repeats, maybe color top consensus repeats by rules, but this should favor certain rules over others
        # FIXME calculate chi-square for each distribution and see if it is different than sum of columns / expected frequency
        # FIXME could just be that there are more kmers that dominate a certain rule.. More kmers originally ID'd for a rule so blasted would yield more... Look at number of kmers for each rule, maybe normalize that way, and compare to overall distribution
        # FIXME do this for each genome and create function to compare distributions


@repeats.command()
@click.option('-s','--subclass_dir',default='./',show_default=True,help='')
def compare_subclasses(subclass_dir):
    """In development: Grab abundance of top classes and subclasses of TEs in each subgenome and compare vs progenitors. Useful to use pairwise similarity function of all consensus repeats."""
    print 'In development'


if __name__ == '__main__':
    repeats()

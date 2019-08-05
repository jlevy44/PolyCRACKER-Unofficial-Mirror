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
def format():
    pass


def create_path(path):
    """Create a path if directory does not exist, raise exception for other errors"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


@format.command(name='maf2bed')
@click.option('-maf', '--last', default='fasta1.fasta2.last,fasta2.fasta1.last', show_default=True, help='Maf output of last alignment. Comma delimited list if multiple maf files.', type=click.Path(exists=False))
@click.option('-w', '--work_dir', default='./', show_default=True, help='Work directory for final outputs.', type=click.Path(exists=False))
def maf2bed(last, work_dir): # FIXME what I can do instead is say that if > 57.5% sequence is covered, than that CDS is one and others are 0, count 1 CDS vs all 0 CDS for identity, does not have to be pure alignment, this should increase similarity scores
    """Convert maf file to bed and perform stats on sequence alignment."""
    from Bio import AlignIO
    #from BCBio import GFF
    import glob
    work_dir += '/'
    last_files = last.split(',')
    final_output = []
    for last in last_files:
        gff_files, bed_files_final = [] , []
        heads = last.split('/')[-1].split('.')[::-1][1:]
        for f_name in heads:
            open(work_dir + f_name+'.gff','w').close()
            gff_files.append(open(work_dir + f_name+'.gff','w'))
            bed_files_final.append(work_dir + f_name+'.bed')
        seqrecs = [[] for i in heads]
        for multiple_alignment in AlignIO.parse(last,'maf'):
            for i,seqrec in enumerate(multiple_alignment): # FIXME
                seqrecs[i].append((seqrec.name,seqrec.annotations['start'] if seqrec.annotations['strand'] == 1 else seqrec.annotations['srcSize'] - seqrec.annotations['start'] - seqrec.annotations['size'], seqrec.annotations['start'] + seqrec.annotations['size'] if seqrec.annotations['strand'] == 1 else seqrec.annotations['srcSize'] - seqrec.annotations['start']))

        #for i, gff_file in enumerate(gff_files):
        #    GFF.write(seqrecs[i],gff_file)
        #    subprocess.call('grep -v "##sequence-region" %s > %s && mv %s %s'%(gff_files_final[i],'temp.gff','temp.gff',gff_files_final[i]),shell=True)
        for i, bed_file in enumerate(bed_files_final):
            pd.DataFrame(seqrecs[i]).to_csv(bed_file, sep='\t',header=None,index=None)
        # FIXME
        fasta_files = []
        last_path = last[:last.rfind('/')+1]
        for f in heads:
            fasta_files.extend(glob.glob(last_path+f+'.fasta') + glob.glob(last_path+f+'.fa'))
        for i,fasta in enumerate(fasta_files):
            Fasta(fasta)
            subprocess.call("awk -v OFS='\\t' '{print $1, 0, $2}' %s > %s"%(fasta+'.fai',fasta+'.bed'),shell=True)
            a = BedTool(fasta+'.bed').sort()
            df = a.intersect(BedTool(bed_files_final[i]).sort().merge()).to_dataframe()
            df2 = a.to_dataframe()
            intersect_sum = (df['end'] - df['start']).sum()
            genome_size = (df2['end'] - df2['start']).sum()
            final_output.append((heads[i],genome_size,intersect_sum,float(intersect_sum)/genome_size))
    df_final = pd.DataFrame(final_output,columns = ['fasta_head','genome_size','length_aligned','percent_aligned'])
    df_final.to_csv(work_dir+'sequence_similarity.csv')
    with open(work_dir+'weighted_sum.txt','w') as f:
        f.write(str((df_final['percent_aligned']*df_final['genome_size']).sum()/float(df_final['genome_size'].sum())))


@format.command(name='convert_mat2R')
@click.option('-npz','--input_matrix',default='clusteringMatrix.npz',help='Input sparse matrix, scipy sparse npz format.',show_default=True, type=click.Path(exists=False))
def convert_mat2R(input_matrix):
    """Convert any sparse matrix into a format to be read by R. Can import matrix into R metagenomics clustering programs."""
    from scipy.io import mmwrite
    mmwrite(input_matrix.replace('.npz','.mtx'),sps.load_npz(input_matrix))


@format.command()
@click.option('-i', '--hipmer_input', default='test.txt', help = 'Input file or directory from hipmer kmer counting run.', show_default=True, type=click.Path(exists=False))
@click.option('-o', '--kcount_output', default='test.final.kcount', help = 'Output kmer count file.', show_default=True, type=click.Path(exists=False))
@click.option('-d', '--run_on_dir', is_flag=True, help='Choose to run on all files in hipmer_input if you have specified a directory for the hipmer input. Directory can only contain hipmer files.')
def hipmer_output_to_kcount(hipmer_input, kcount_output, run_on_dir):
    """Converts hipmer kmer count output into a kmer count, kcount, file."""
    if run_on_dir:
        hipmer_path = hipmer_input + '/'
        subprocess.call("cat %s | awk '{OFS = \"\\t\"; sum=0; for (i=2; i<=7; i++) { sum+= $i }; if (sum >= 3) print $1, sum}' > %s"%(' '.join([hipmer_path+hipmer_input for hipmer_input in os.listdir(hipmer_path)]),kcount_output),shell=True)
    else:
        subprocess.call("cat %s | awk '{OFS = \"\\t\"; sum=0; for (i=2; i<=7; i++) { sum+= $i }; if (sum >= 3) print $1, sum}' > %s"%(hipmer_input,kcount_output),shell=True)


@format.command()
@click.option('-a', '--anchor_file', help = 'Lifted anchor file generated from basic synteny run using jcvi tools.', type=click.Path(exists=False))
@click.option('-q', '--qbed', help='First bed file.', type=click.Path(exists=False))
@click.option('-s', '--sbed', help='Second bed file.', type=click.Path(exists=False))
def anchor2bed(anchor_file, qbed, sbed):
    """Convert syntenic blocks of genes to bed coordinates between the two genomes being compared."""
    with open(anchor_file,'r') as f:
        anchors = f.read().split('###')
    with open(qbed,'r') as f:
        qbed = {}
        for line in f:
            if line:
                lineL = line.split()
                qbed[lineL[3]] = [lineL[0]] + map(int,lineL[1:3])
    #print qbed
    with open(sbed,'r') as f:
        sbed = {}
        for line in f:
            if line:
                lineL = line.split()
                sbed[lineL[3]] = [lineL[0]] + map(int,lineL[1:3])
    with open(anchor_file.replace('.lifted.anchors','.bed'),'w') as f:
        for anchor in anchors:
            if anchor:
                #print anchor
                q_coords = []
                s_coords = []
                for line in anchor.splitlines():
                    if line:
                        genes = line.split()[:2]
                        #print genes
                        q_coords.append(qbed[genes[0]])
                        s_coords.append(sbed[genes[1]])
                #print q_coords
                q_coords = pd.DataFrame(np.array(q_coords)).sort_values([0,1]).as_matrix()
                s_coords = pd.DataFrame(np.array(s_coords)).sort_values([0,1]).as_matrix()
                f.write('\t'.join(map(str,[q_coords[0,0],q_coords[:,1:].min(),q_coords[:,1:].max(), s_coords[0,0],s_coords[:,1:].min(),s_coords[:,1:].max()]))+'\n')
    with open(anchor_file.replace('.lifted.anchors','.bed'),'r') as f:
        links = np.array([line.split() for line in f.read().splitlines()])
        colors_set = {color:i+1 for i, color in enumerate(set(links[:,0]))}
        colors = pd.DataFrame(np.vectorize(lambda color: colors_set[color])(links[:,0]),columns=['Color'])
        colors.to_csv('link_colors.csv',index=False)
        links = pd.DataFrame(links,columns=['seg1','start1','end1','seg2','start2','end2'])
        links.to_csv('links.csv',index=False)
        # FIXME, need to grab correct orientation!!!


if __name__ == '__main__':
    format()
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
def polycracker():
    pass


def create_path(path):
    """Create a path if directory does not exist, raise exception for other errors"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


@polycracker.command(name='run_pipeline')
@click.option('-m','--memory', default = '47', help='Amount of memory to use on sge or slurm submission job, in Gigabytes.', show_default=True)
@click.option('-t','--time', default = '11', help='Number of hours to run sge or slurm submission job', show_default = True)
@click.option('-N','--nodes', default = '1', help='Number of nodes for slurm submission job', show_default = True)
@click.option('-b','--batch_script', default = 'runCluster.sh', help = 'Run custom batch script. Default will run the pipeline.', show_default = True)
@click.option('-J','--job_name', default = 'polyCRACKER', help = 'Custom name for job.', show_default=True)
@click.option('-n','--no_options', is_flag=True, help= 'Run slurm submission script with no options.')
@click.option('-a','--account_name', default='fungalp', help= 'Account name for running jobs.', show_default=True)
@click.option('-nh','--no_nohup', is_flag=True, help= 'No nohup for pipeline run.')
def run_pipeline(memory, time, nodes, batch_script, job_name, no_options, account_name, no_nohup):
    """Run polycracker pipeline."""
    subprocess.call('%ssh %s %s'%('nohup ' if not no_nohup else '', batch_script, '' if no_nohup else '&'),shell=True)
    submit_txt = 'nohup sh %s &\n'%batch_script
    run_files = [file for file in os.listdir('.') if file.startswith('submission_history.') and file.endswith('.txt')]
    with open('submission_history.%i.txt'%(len(run_files)),'w') as f:
        f.write(submit_txt + '\n'.join(['--memory '+memory, '--time '+time,'--nodes '+nodes,
                                        '--batch_script '+batch_script,'--job_name '+job_name, '--no_options '+str(no_options), '--account_name '+account_name]))


@polycracker.command(name='test_pipeline')
@click.option('-o', '--output_folder', default = 'test_results', help='folder where the output should be copied.', type=click.Path(exists=False))
@click.option('-m','--memory', default = '47', help='Amount of memory to use on sge or slurm submission job, in Gigabytes.', show_default=True)
@click.option('-t','--time', default = '11', help='Number of hours to run sge or slurm submission job', show_default = True)
@click.option('-N','--nodes', default = '1', help='Number of nodes for slurm submission job', show_default = True)
def test_pipeline(output_folder, memory, time, nodes):
    genome_path = os.path.join(output_folder, 'extracted_subgenomes')
    create_path(genome_path)
    plots_path = os.path.join(output_folder, 'plots')
    create_path(plots_path)
    stats_path = os.path.join(output_folder, 'test_statistics')
    create_path(stats_path)
    create_path('analysisOutputs')
    subprocess.call('scp test_data/config_polyCRACKER.txt .', shell=True)
    subprocess.call('polycracker run_pipeline -m %s -t %s -N %s -nh'%(memory, time, nodes), shell=True)
    # subprocess.call('python polycracker.py subgenomeExtraction -p ./test_data/test_fasta_files/ -s analysisOutputs/SpectralClusteringmain_tsne_2_n3/bootstrap_0/ -os analysisOutputs/SpectralClusteringmain_tsne_2_n3/ -g algae_split.fa -go algae_split.fa -l 26 -b 2 -kl 30 -dk 10 -kv 1 -u 8,2', shell=True)
    subprocess.call('polycracker final_stats -s -dict subgenome_0:Creinhardtii,subgenome_1:Csubellipsoidea -pbed analysisOutputs/SpectralClusteringmain_tsne_2_n3/clusterResults/', shell=True)
    subprocess.call('polycracker convert_subgenome_output_to_pickle -id analysisOutputs/SpectralClusteringmain_tsne_2_n3/clusterResults/ -s analysisOutputs/SpectralClusteringmain_tsne_2_n3/scaffolds_connect.p', shell=True)
    subprocess.call('scp analysisOutputs/SpectralClusteringmain_tsne_2_n3/bootstrap_0/model_subgenome_*.fa ./%s/extracted_subgenomes' % output_folder, shell=True)
    subprocess.call('scp polycracker.stats.analysis.csv ./%s/test_statistics' % output_folder, shell=True)
    subprocess.call('scp SpectralClusteringmain_tsne_2_n3ClusterTest.html ./%s/plots' % output_folder, shell=True)
    subprocess.call('polycracker number_repeatmers_per_subsequence && scp *.png ./%s/plots' % output_folder, shell=True)
    subprocess.call('polycracker plotPositions -npy analysisOutputs/SpectralClusteringmain_tsne_2_n3/graphInitialPositions.npy -p analysisOutputs/SpectralClusteringmain_tsne_2_n3/scaffolds_connect.p -o ./%s/plots/spectral_graph.html -npz analysisOutputs/SpectralClusteringmain_tsne_2_n3/spectralGraph.npz -l spectral' % output_folder, shell=True)
    click.echo('Please see results in ./%s.' % output_folder)
    click.echo('\nOriginal genome in ./test_data/test_fasta_files .\n')


@polycracker.command(name='splitFasta')
@click.option('-i', '--fasta_file', help='Input polyploid fasta filename.', type=click.Path(exists=False))
@click.option('-d', '--fasta_path', default='./fasta_files/', help='Directory containing polyploids fasta file', show_default=True, type=click.Path(exists=False))
@click.option('-c', '--chunk_size', default=75000, help='Length of chunk of split fasta file', show_default=True)
@click.option('-p', '--prefilter', default=0, help='Remove chunks based on certain conditions as a preprocessing step to reduce memory', show_default=True)
@click.option('-m', '--min_chunk_size', default=0, help='If min_chunk_threshold and prefilter flag is on, use chunks with lengths greater than a certain size', show_default=True)
@click.option('-r', '--remove_non_chunk', default=0, help='If min_chunk_threshold and prefilter flag is on, remove chunks that are not the same size as chunk_size', show_default=True)
@click.option('-t', '--min_chunk_threshold', default=0, help='If prefilter is on, remove chunks with lengths less than a certain threshold', show_default=True)
@click.option('-l', '--low_memory', default=0, help='Must use with prefilter in order for prefilter to work.', show_default=True)
def splitFasta(fasta_file, fasta_path, chunk_size, prefilter, min_chunk_size, remove_non_chunk, min_chunk_threshold, low_memory):
    """Split fasta file into chunks of a specified length"""
    global inputStr, key
    if not os.path.exists(fasta_path):
        raise OSError('Specified fasta_file directory does not exist')
    Fasta(fasta_path+fasta_file)
    faiFile = fasta_path+fasta_file + '.fai'
    def grabLine(positions):
        global inputStr
        global key
        if positions[-1] != 'end':
            return '%s\t%s\t%s\t%s\n'%tuple([key]+map(str,positions[0:2])+['_'.join([key]+map(str,positions[0:2]))])
        else:
            return '%s\t%s\t%s\t%s\n' % (key, str(positions[0]), str(inputStr), '_'.join([key, str(positions[0]), str(inputStr)]))
    def split(inputStr=0,width=60):
        if inputStr:
            positions = np.arange(0,inputStr,width)
            posFinal = []
            if inputStr > width:
                for i in range(len(positions[:-1])):
                    posFinal += [(positions[i],positions[i+1])]
            posFinal += [(positions[-1],'end')]
            splitLines = map(grabLine,posFinal)
            return splitLines
        else:
            return ''
    bedText = []
    for key, seqLength in [tuple(line.split('\t')[0:2]) for line in open(faiFile,'r') if line]:
        inputStr = int(seqLength)
        bedText += split(inputStr,chunk_size)
    with open('correspondence.bed','w') as f:
        f.write('\n'.join(bedText))
    corrBed = BedTool('correspondence.bed').sort()
    corrBed.saveas('correspondence.bed')
    corrBed.saveas('correspondenceOriginal.bed')
    if prefilter and low_memory:
        if remove_non_chunk:
            scaffolds = corrBed.filter(lambda x: len(x) == chunk_size)
            scaffolds.saveas('correspondence.bed')
            oldLines = corrBed.filter(lambda x: len(x) != chunk_size)
            oldLines.saveas('correspondence_filteredOut.bed')
            subprocess.call('bedtools getfasta -fi %s -fo %s -bed %s -name'%(fasta_path+fasta_file,fasta_path + 'filteredOut.fa','correspondence_filteredOut.bed'),shell=True)
        elif min_chunk_threshold:
            scaffolds = corrBed.filter(lambda x: len(x) >= min_chunk_size)
            scaffolds.saveas('correspondence.bed')
            oldLines = corrBed.filter(lambda x: len(x) < min_chunk_size)
            oldLines.saveas('correspondence_filteredOut.bed')
            subprocess.call('bedtools getfasta -fi %s -fo %s -bed %s -name'%(fasta_path+fasta_file,fasta_path + 'filteredOut.fa','correspondence_filteredOut.bed'),shell=True)
    subprocess.call('bedtools getfasta -fi %s -fo %s -bed %s -name'%(fasta_path+fasta_file,fasta_path+fasta_file.split('.fa')[0]+'_split.fa','correspondence.bed'),shell=True)
    Fasta((fasta_path+fasta_file).split('.fa')[0]+'_split.fa')


@polycracker.command(name='writeKmerCount')
@click.option('-d', '--fasta_path', default='./fasta_files/', help='Directory containing chunked polyploids fasta file', show_default=True, type=click.Path(exists=False))
@click.option('-k', '--kmercount_path', default='./kmercount_files', help='Directory containing kmer count files', show_default=True, type=click.Path(exists=False))
@click.option('-l', '--kmer_length', default='23,31', help='Length of kmers to find; can include multiple lengths if comma delimited (e.g. 23,25,27)', show_default=True)
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
def writeKmerCount(fasta_path, kmercount_path, kmer_length, blast_mem):
    """Takes list of fasta files and runs kmercountexact.sh to find how many of each kmer of specified length is in genome"""
    kmer_lengths = kmer_length.split(',')
    blast_memStr = "export _JAVA_OPTIONS='-Xmx%sG'"%(blast_mem)
    print(blast_memStr)
    create_path(kmercount_path)
    for fastaFile in os.listdir(fasta_path):
        if (fastaFile.endswith('.fa') or fastaFile.endswith('.fasta')) and '_split' in fastaFile:
            kmerFiles = []
            f = fastaFile.rstrip()
            print f
            outFileNameFinal = f[:f.rfind('.')] + '.kcount'
            open(kmercount_path + '/' + outFileNameFinal, 'w').close()
            for kmerL in kmer_lengths:
                scriptName = f[:f.rfind('.')] + kmerL + '.sh'
                outFileName = f[:f.rfind('.')]+kmerL+'.kcount'
                kmerFiles.append(kmercount_path+'/'+outFileName)
                lineOutputList = [fasta_path, fastaFile, kmercount_path, outFileName, kmerL]
                if int(kmerL) <= 31:
                    bbtoolsI = open(scriptName, 'w')
                    bbtoolsI.write('#!/bin/bash\n'+blast_memStr+'\nkmercountexact.sh overwrite=true fastadump=f mincount=3 in=%s/%s out=%s/%s k=%s -Xmx%sg\n' %tuple(lineOutputList+[blast_mem]))
                    bbtoolsI.close()
                else:
                    with open(scriptName,'w') as f2:
                        f2.write('#!/bin/bash\njellyfish count -m %s -s %d -t 15 -C -o %s/mer_counts.jf %s/%s && jellyfish dump %s/mer_counts.jf -c > %s/%s'%(kmerL,os.stat(fasta_path+'/'+fastaFile).st_size,kmercount_path,fasta_path,fastaFile,kmercount_path,kmercount_path,outFileName))
                try:
                    subprocess.call('sh %s' % scriptName, shell=True)
                except:
                    print 'Unable to run %s via command line..' % outFileName
            subprocess.call('cat %s > %s'%(' '.join(kmerFiles), kmercount_path + '/' + outFileNameFinal), shell=True)
            subprocess.call('rm %s'%(' '.join(kmerFiles)), shell=True)


@polycracker.command(name='kmer2Fasta')
@click.option('-k', '--kmercount_path', default='./kmercount_files/', help='Directory containing kmer count files', show_default=True, type=click.Path(exists=False))
@click.option('-lc', '--kmer_low_count', default=100, help='Omit kmers from analysis that have less than x occurrences throughout genome.', show_default=True)
@click.option('-hb', '--high_bool', default=0, help='Enter 1 if you would like to use the kmer_high_count option.', show_default=True)
@click.option('-hc', '--kmer_high_count', default=2000000, help='If high_bool is set to one, omit kmers that have greater than x occurrences.', show_default=True)
@click.option('-s', '--sampling_sensitivity', default=1, help='If this option, x, is set greater than one, kmers are sampled at lower frequency, and total number of kmers included in the analysis is reduced to (total #)/(sampling_sensitivity), after threshold filtering', show_default=True)
def kmer2Fasta(kmercount_path, kmer_low_count, high_bool, kmer_high_count, sampling_sensitivity):
    """Converts kmer count file into a fasta file for blasting, after some threshold filtering and sampling"""
    if sampling_sensitivity < 1:
        sampling_sensitivity = 1
    count = 0
    for kmer in os.listdir(kmercount_path):
        if kmer.endswith('.kcount'):
            with open(kmercount_path+kmer,'r') as f, open(kmercount_path+kmer+'.fa','w') as f2:
                if high_bool:
                    for line in f:
                        lineList = line.split() # line.split('\t')
                        if line and int(lineList[-1]) >= kmer_low_count and int(lineList[-1]) <= kmer_high_count:
                            if count % sampling_sensitivity == 0:
                                f2.write('>%s\n%s\n'%tuple([lineList[0]]*2))
                            count += 1
                else:
                    for line in f:
                        if line and int(line.split()[-1]) >= kmer_low_count: # line.split('\t')
                            if count % sampling_sensitivity == 0:
                                f2.write('>%s\n%s\n'%tuple([line.split()[0]]*2)) # '\t'
                            count += 1


@polycracker.command(name='blast_kmers')
@click.option('-m', '--blast_mem', default='48', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
@click.option('-r', '--reference_genome', help='Genome to blast against.', type=click.Path(exists=False))
@click.option('-k', '--kmer_fasta', help='Kmer fasta converted from kmer count file', type=click.Path(exists=False))
@click.option('-o', '--output_file', default = 'results.sam', show_default=True, help='Output sam file.', type=click.Path(exists=False))
@click.option('-kl', '--kmer_length', default=13, help='Kmer length for mapping.', show_default=True)
@click.option('-pm', '--perfect_mode', default = 1, help='Perfect mode.', show_default=True)
@click.option('-t', '--threads', default = 5, help='Threads to use.', show_default=True)
def blast_kmers(blast_mem, reference_genome, kmer_fasta, output_file, kmer_length, perfect_mode, threads):
    """Maps kmers fasta file to a reference genome."""
    blast_memStr = "export _JAVA_OPTIONS='-Xms5G -Xmx%sG'"%(blast_mem)
    subprocess.call('%s && bbmap.sh vslow=t ambiguous=all noheader=t secondary=t k=%d perfectmode=%s threads=%s maxsites=2000000000 outputunmapped=f ref=%s in=%s outm=%s -Xmx%sg'%(blast_memStr,kmer_length,'t' if perfect_mode else 'f', threads, reference_genome, kmer_fasta, output_file, blast_mem), shell=True)


@polycracker.command(name='blast2bed')
@click.option('-i', '--blast_file', default='./blast_files/results.sam', help='Output file after blasting kmers to genome. Must be input into this command.', show_default=True, type=click.Path(exists=False))
@click.option('-b', '--bb', default=1, help='Whether bbtools were used in generating blasted sam file.', show_default=True)
@click.option('-l', '--low_memory', default=0, help='Do not merge the bed file. Takes longer to process but uses lower memory.', show_default=True)
@click.option('-o', '--output_bed_file', default = 'blasted.bed', show_default=True, help='Output bed file, not merged.', type=click.Path(exists=False))
@click.option('-om', '--output_merged_bed_file', default = 'blasted_merged.bed', show_default=True, help='Output bed file, merged.', type=click.Path(exists=False))
@click.option('-ex', '--external_call', is_flag=True, help='External call from non-polyCRACKER pipeline.')
def blast2bed(blast_file, bb , low_memory, output_bed_file, output_merged_bed_file, external_call):
    """Converts the blast results from blast or bbtools into a bed file"""
    if external_call:
        subprocess.call("awk -v OFS='\\t' '{ print $3, 0, 100, $1 }' %s > %s"%(blast_file,output_bed_file),shell=True)
    else:
        if bb:
            with open(blast_file,'r') as f, open(output_bed_file,'w') as f2:
                for line in f:
                    if line:
                        l1 = line.split('\t')[2].split('::')[0]
                        f2.write('\t'.join([l1] + ['0', str(int(l1.split('_')[-1]) - int(l1.split('_')[-2]))] + [line.split('\t')[0]]) + '\n')
        else:
            with open(blast_file,'r') as f, open(output_bed_file,'w') as f2:
                for line in f:
                    if line:
                        l1 = line.split('\t')[1].split('::')[0]
                        f2.write('\t'.join([l1] + ['0',str(int(l1.split('_')[-1])-int(l1.split('_')[-2]))] + [line.split('\t')[0]])+'\n')
    if low_memory == 0:
        b = pybedtools.BedTool(output_bed_file).sort().merge(c=4,o='collapse')
        b.saveas(output_merged_bed_file)


def findScaffolds():
    with open('correspondence.bed','r') as f:
        lineList = sorted([line.split('\t')[-1].strip('\n') for line in f.readlines() if line])
    return lineList


def findKmerNames(kmercount_path, genome):
    for file in os.listdir(kmercount_path):
        if file.startswith(genome[:genome.find('.fa')]) and (file.endswith('.fa') or file.endswith('.fasta')):
            print file
            with open(kmercount_path + file,'r') as f:
                listKmer = sorted([line.strip('\n') for line in f.readlines()[1::2] if line])
            return listKmer


@polycracker.command(name='generate_Kmer_Matrix')
@click.option('-d', '--kmercount_path', default='./kmercount_files/', help='Directory containing kmer count files', type=click.Path(exists=False))
@click.option('-g', '--genome', help='File name of chunked genome fasta', type=click.Path(exists=False))
@click.option('-c', '--chunk_size', default=75000, help='Length of chunk of split/chunked fasta file', show_default=True)
@click.option('-mc', '--min_chunk_size', default=0, help='If min_chunk_threshold and prefilter flag is on, use chunks with lengths greater than a certain size', show_default=True)
@click.option('-r', '--remove_non_chunk', default=1, help='If min_chunk_threshold and prefilter flag is on, remove chunks that are not the same size as chunk_size', show_default=True)
@click.option('-t', '--min_chunk_threshold', default=0, help='If prefilter is on, remove chunks with lengths less than a certain threshold', show_default=True)
@click.option('-l', '--low_memory', default=0, help='Must use with prefilter in order for prefilter to work. Without prefilter, populates matrix using unmerged blasted bed (slow).', show_default=True)
@click.option('-p', '--prefilter', default=0, help='If selected, will now add chunks that were removed during preprocessing.', show_default=True)
@click.option('-f', '--fasta_path', default='./fasta_files/', help='Directory containing chunked polyploids fasta file', type=click.Path(exists=False))
@click.option('-gs', '--genome_split_name', help='File name of chunked genome fasta', type=click.Path(exists=False))
@click.option('-go', '--original_genome', help='File name of original, prechunked genome fasta', type=click.Path(exists=False))
@click.option('-i', '--input_bed_file', default = 'blasted.bed', show_default=True, help='Input bed file, not merged.', type=click.Path(exists=False))
@click.option('-im', '--input_merged_bed_file', default = 'blasted_merged.bed', show_default=True, help='Input bed file, merged.', type=click.Path(exists=False))
@click.option('-npz', '--output_sparse_matrix', default = 'clusteringMatrix.npz', show_default=True, help='Output sparse matrix file, in npz format.', type=click.Path(exists=False))
@click.option('-k', '--output_kmer_file', default = 'kmers.p', show_default=True, help='Output kmer pickle file.', type=click.Path(exists=False))
@click.option('-tf', '--tfidf', default=0, help='If set to one, no sequence length normalization will be done.', show_default=True)
def generate_Kmer_Matrix(kmercount_path, genome, chunk_size, min_chunk_size, remove_non_chunk, min_chunk_threshold, low_memory, prefilter, fasta_path, genome_split_name, original_genome, input_bed_file, input_merged_bed_file, output_sparse_matrix, output_kmer_file, tfidf):
    """From blasted bed file, where kmers were blasted to the polyploid genome, now generate a sparse matrix that contains information pertaining to the distribution of kmers (columns) for a given region of the genome (rows)"""
    if prefilter and low_memory:
        subprocess.call('bedtools getfasta -fi %s -fo %s -bed %s -name'%(fasta_path+original_genome,fasta_path+genome_split_name,'correspondenceOriginal.bed'),shell=True)
    kmers = findKmerNames(kmercount_path, genome)
    originalScaffolds = findScaffolds()
    if remove_non_chunk and prefilter == 0:
        scaffolds = list(filter(lambda scaffold: abs(int(scaffold.split('_')[-1]) - int(scaffold.split('_')[-2])) == chunk_size,originalScaffolds))
    elif min_chunk_threshold and prefilter == 0:
        scaffolds = list(filter(lambda scaffold: abs(int(scaffold.split('_')[-1]) - int(scaffold.split('_')[-2])) >= min_chunk_size,originalScaffolds))
    else:
        scaffolds = originalScaffolds
    scaffoldIdx = {scaffold:i for i,scaffold in enumerate(scaffolds)}
    kmerIdx = {kmer:i for i,kmer in enumerate(kmers)}
    data = sps.dok_matrix((len(scaffolds), len(kmers)),dtype=np.float32)
    scaffoldL = np.array([map(float, scaffold.split('_')[-2:]) for scaffold in scaffolds])
    scaffoldLengths = abs(scaffoldL[:, 1] - scaffoldL[:, 0]) / 5000.
    if low_memory:
        with open(input_bed_file, 'r') as f:
            for line in f:
                if line:
                    try:
                        lineL = line.strip('\n').split('\t')
                        data[scaffoldIdx[lineL[0]], kmerIdx[lineL[-1]]] += 1.
                    except:
                        pass
    else:
        with open(input_merged_bed_file, 'r') as f:
            for line in f:
                if line:
                    listLine = line.rstrip('\n').split('\t')
                    if listLine[0] in scaffolds:
                        counts = Counter(listLine[-1].split(','))
                        for key in counts:
                            try:
                                data[scaffoldIdx[listLine[0]], kmerIdx[key]] = counts[key]
                            except:
                                pass
    del scaffoldIdx, kmerIdx
    data = data.tocsc()
    # divide every row by scaffold length
    if not tfidf:
        for i in range(len(scaffoldLengths)):
            data[i, :] /= scaffoldLengths[i]
    sps.save_npz(output_sparse_matrix, data)
    with open('rowNames.txt', 'w') as f:
        f.write('\n'.join('\t'.join([str(i), scaffolds[i]]) for i in range(len(scaffolds))))
    with open('colNames.txt', 'w') as f:
        f.write('\n'.join('\t'.join([str(i), kmers[i]]) for i in range(len(kmers))))
    pickle.dump(scaffolds,open('scaffolds.p','wb'),protocol=2)
    pickle.dump(originalScaffolds, open('originalScaffolds.p', 'wb'), protocol=2)
    pickle.dump(kmers,open(output_kmer_file,'wb'),protocol=2)


@polycracker.command(name='transform_plot')
@click.option('-t', '--technique', default='kpca', help='Dimensionality reduction technique to use.', type=click.Choice(['kpca','factor','feature', 'lda', 'nmf', 'tsne']), show_default=True)
@click.option('-s', '--n_subgenomes', default = 2, help='Number of subgenomes', show_default=True)
@click.option('-m', '--metric', default='cosine', help='Kernel for KPCA if kpca technique is chosen. Cosine is particularly effective.', type=click.Choice(['linear','poly','rbf','sigmoid','cosine']), show_default=True)
@click.option('-d', '--n_dimensions', default=3, help='Number of dimensions to reduce to.', show_default=True)
@click.option('-tf', '--tfidf', default=0, help='If set to one, tfidf normalization will be used instead of standard scaling.', show_default=True)
def transform_plot(technique, n_subgenomes, metric, n_dimensions, tfidf):
    """Perform dimensionality reduction on a sparse matrix and plot the results."""
    if n_subgenomes < 2:
        n_subgenomes = 2
    if n_dimensions < 3:
        n_dimensions = 3
    peak = 'main'
    if os.path.exists(peak + '_' + technique + '_%d'%(n_subgenomes) + 'Reduction.html') == 0:
        data = sps.load_npz('clusteringMatrix.npz')
        scaffolds = pickle.load(open('scaffolds.p', 'rb'))
        N = n_dimensions
        dimensionalityReducers = {'kpca': KernelPCA(n_components=N,kernel=metric,random_state=RANDOM_STATE), 'factor': FactorAnalysis(n_components=N),
                                  'feature': FeatureAgglomeration(n_clusters=N), 'lda': LatentDirichletAllocation(n_topics=N), 'nmf': NMF(n_components=N)}
        if technique == 'tsne':
            from MulticoreTSNE import MulticoreTSNE as TSNE
            dimensionalityReducers.update({'tsne': TSNE(n_components=n_dimensions,n_jobs=6,metric=metric if metric == 'cosine' else 'euclidean',learning_rate=200,perplexity=30,angle=0.5,random_state=RANDOM_STATE)})
        if not tfidf:
            data = StandardScaler(with_mean=False).fit_transform(data)
        else:
            from sklearn.feature_extraction.text import TfidfTransformer
            data = TfidfTransformer().fit_transform(data)
        if technique not in 'kpca':
            data = KernelPCA(n_components=25,random_state=RANDOM_STATE).fit_transform(data)
        transformed_data = dimensionalityReducers[technique].fit_transform(data)
        np.save('%s_%s_%d_transformed3D.npy'%(peak,technique,n_subgenomes), transformed_data)
        if n_dimensions > 3:
            metric = 'linear'
            transformed_data = KernelPCA(n_components=3,kernel=metric,random_state=RANDOM_STATE).fit_transform(transformed_data)
        plots = []
        plots.append(
            go.Scatter3d(x=transformed_data[:, 0], y=transformed_data[:, 1], z=transformed_data[:, 2], name='Data',
                         mode='markers',
                         marker=dict(color='b', size=2), text=scaffolds))
        fig = go.Figure(data=plots)
        py.plot(fig, filename=peak + '_' + technique + '_%d'%(n_subgenomes) + 'Reduction.html', auto_open=False)
    else:
        subprocess.call('touch %s'%(peak + '_' + technique + '_%d'%(n_subgenomes) + 'Reduction.html'),shell=True)

@polycracker.command(name='reset_transform')
def reset_transform():
    """Remove all html files from main work directory. Must do this if hoping to retransform sparse data."""
    subprocess.call('rm *.html',shell=True)


# begin clustering
class GeneticHdbscan:

    def __init__(self,metric='euclidean',min_clusters=3,max_cluster_size=1000,max_samples=500,validity_measure='silhouette', generations_number=8, gene_mutation_prob=0.45, gene_crossover_prob = 0.45, population_size=50, interval=10, upper_bound=False, verbose=True):
        self.hdbscan_metric = (metric if metric not in ['ectd','cosine','mahalanobis'] else 'precomputed')
        self.min_clusters = min_clusters
        self.max_cluster_size = max_cluster_size
        self.max_samples = max_samples
        self.validity_measure = validity_measure
        self.generations_number = generations_number
        self.gene_mutation_prob = gene_mutation_prob
        self.gene_crossover_prob = gene_crossover_prob
        self.population_size = population_size
        self.scoring_method = lambda X, y: hdbscan.validity.validity_index(X,y,metric=self.hdbscan_metric) if self.validity_measure == 'hdbscan_validity' else (silhouette_score(X,y,metric='precomputed' if metric =='mahalanobis' else 'mahalanobis',V=(np.cov(X,rowvar=False) if metric != 'mahalanobis' else '')) if self.validity_measure == 'silhouette' else calinski_harabaz_score(X,y))
        self.low_counts = (2,2)
        self.interval = interval
        self.upper_bound = upper_bound
        self.verbosity = verbose

    def cluster_data(self, X, min_cluster_size, min_samples, cluster_selection_method):
        return hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples= min_samples, cluster_selection_method= cluster_selection_method, metric = self.hdbscan_metric, alpha = 1.0).fit_predict(X)

    def return_cluster_score(self, X, min_cluster_size, min_samples, cluster_selection_method):
        labels = self.cluster_data(X, min_cluster_size, min_samples, cluster_selection_method) # , n_neighbors
        n_clusters = labels.max() + 1
        # print labels
        X = X if self.validity_measure == 'hdbscan_validity' else X[labels != -1,:]
        y = labels if self.validity_measure == 'hdbscan_validity' else labels[labels != -1]
        # print y
        if list(y):
            return self.scoring_method(X,y)/(((1. + (abs(n_clusters - self.min_clusters) if self.upper_bound else 0.)) if n_clusters >= self.min_clusters else float(self.min_clusters - n_clusters + 1))*(1.+len(labels[labels == -1])/float(len(labels))))
        else:
            return 0

    def fit(self, X):
        best_params, best_score, score_results, hist, log = maximize(self.return_cluster_score, dict(min_cluster_size = np.unique(np.linspace(self.low_counts[0],self.max_cluster_size,self.interval).astype(int)).tolist(), min_samples = np.unique(np.linspace(self.low_counts[1],self.max_samples, self.interval).astype(int)).tolist(), cluster_selection_method= ['eom'] ), dict(X=X), verbose=self.verbosity, n_jobs = 1, generations_number=self.generations_number, gene_mutation_prob=self.gene_mutation_prob, gene_crossover_prob = self.gene_crossover_prob, population_size = self.population_size) # fixme, 'leaf' # n_neighbors = np.unique(np.linspace(low_counts[2], n_neighbors, 10).astype(int)).tolist()),
        self.labels_ = self.cluster_data(X,min_cluster_size=best_params['min_cluster_size'], min_samples=best_params['min_samples'], cluster_selection_method= best_params['cluster_selection_method'])


@polycracker.command(name='find_best_cluster_parameters')
@click.pass_context
@click.option('-f', '--file', help='Numpy .npy file containing positions after dimensionality reduction')
@click.option('-y', '--y_true_pickle', default='colors_pickle.p', show_default=True, help='Pickle file containing ground truths.')
@click.option('-s', '--n_subgenomes', default = 3, help='Number of subgenomes', show_default=True)
@click.option('-g', '--generations_number', default=10, show_default=True, help='Number of generations.')
@click.option('-gm', '--gene_mutation_prob', default=0.45, show_default=True, help='Gene mutation probability.')
@click.option('-gc', '--gene_crossover_prob', default=0.45, show_default=True, help='Gene crossover probability.')
@click.option('-p', '--population_size', default=250, show_default=True, help='Population size.')
def find_best_cluster_parameters(ctx,file,y_true_pickle,n_subgenomes,generations_number, gene_mutation_prob, gene_crossover_prob, population_size):
    """In development: Experimenting with genetic algorithm (GA). Use GA to find best cluster method / hyperparameters given ground truth. These hyperparameter scans can help us find out what is important in finding good clusters rather than trial and error."""
    from sklearn.metrics import fowlkes_mallows_score  # v_measure_score
    try:
        os.makedirs('cluster_tests/')
    except:
        pass
    y_true = pickle.load(open(y_true_pickle,'r'))
    cluster_function = lambda file, cluster_method, n_subgenomes, metric, n_neighbors, weighted_nn, grab_all, gamma, min_span_tree: ctx.invoke(cluster,file=file, cluster_method=cluster_method, n_subgenomes=n_subgenomes, metric=metric, n_neighbors=n_neighbors, weighted_nn=weighted_nn, grab_all=grab_all, gamma=gamma, min_span_tree=min_span_tree)
    subprocess.call('rm GA_scores.txt',shell=True)
    def cluster_scoring(file, cluster_method, n_subgenomes, metric, n_neighbors, weighted_nn, grab_all, gamma, min_span_tree):
        os.system('rm *.html &> /dev/null')
        try:
            y_pred = cluster_function(file, cluster_method, n_subgenomes, metric, n_neighbors, weighted_nn, grab_all, gamma, min_span_tree)
            os.system('mv *.html cluster_tests/ &> /dev/null')
            #print(y_true, y_pred)
            score = fowlkes_mallows_score(y_true,y_pred)#v_measure_score(y_true,y_pred)
            click.echo(str(dict(zip(['cluster_method', 'n_subgenomes', 'metric', 'n_neighbors', 'weighted_nn', 'grab_all', 'gamma', 'min_span_tree','score'],[cluster_method, n_subgenomes, metric, n_neighbors, weighted_nn, grab_all, gamma, min_span_tree,score]))))
            #f.write(str(dict(zip(['cluster_method', 'n_subgenomes', 'metric', 'n_neighbors', 'weighted_nn', 'grab_all', 'gamma', 'min_span_tree','score'],[cluster_method, n_subgenomes, metric, n_neighbors, weighted_nn, grab_all, gamma, min_span_tree,score]))) + '\n')
            return score
        except:
            #f.write(str(dict(zip(['cluster_method', 'n_subgenomes', 'metric', 'n_neighbors', 'weighted_nn', 'grab_all', 'gamma', 'min_span_tree','score'],[cluster_method, n_subgenomes, metric, n_neighbors, weighted_nn, grab_all, gamma, min_span_tree,0]))) + '\n')
            click.echo(str(dict(zip(['cluster_method', 'n_subgenomes', 'metric', 'n_neighbors', 'weighted_nn', 'grab_all', 'gamma', 'min_span_tree','score'],[cluster_method, n_subgenomes, metric, n_neighbors, weighted_nn, grab_all, gamma, min_span_tree,0]))))
            return 0
    best_params, best_score, score_results, hist, log = maximize(cluster_scoring,dict(cluster_method=['SpectralClustering','KMeans','GMM','BGMM','hdbscan_genetic'],
                                                                                      metric=['cityblock','cosine','euclidean','l1','l2','manhattan','braycurtis','canberra','chebyshev','correlation','dice','hamming','mahalanobis','matching','minkowski','rogerstanimoto','russellrao','jaccard','yule','sokalsneath'], #,'seuclidean','sokalmichener',,'sqeuclidean' ,'jaccard','kulsinski'
                                                                                      n_neighbors=range(3,100,20)),dict(file=file,n_subgenomes=n_subgenomes,min_span_tree=1,grab_all=1,weighted_nn=0,gamma=1),verbose=True, n_jobs = 1, generations_number=generations_number, gene_mutation_prob=gene_mutation_prob, gene_crossover_prob = gene_crossover_prob, population_size = population_size) #,weighted_nn = [0,1],grab_all=[0,1],gamma=np.logspace(-4,100,50),min_span_tree=[0,1]
    click.echo(str(best_params))
    with open('cluster_tests.txt','w') as f:
        f.write(str(best_params)+'\n\n'+str(score_results))


@polycracker.command(name='cluster')
@click.option('-f', '--file', help='Numpy .npy file containing positions after dimensionality reduction')#, type=click.Path(exists=False))
@click.option('-c', '--cluster_method', default='SpectralClustering', help='Clustering method to use.', type=click.Choice(['SpectralClustering','KMeans','GMM','BGMM','hdbscan_genetic']), show_default=True)
@click.option('-s', '--n_subgenomes', default = 2, help='Number of subgenomes', show_default=True)
@click.option('-m', '--metric', default='cosine', help='Distance metric used to compute affinity matrix, used to find nearest neighbors graph for spectral clustering.', type=click.Choice(['cityblock','cosine','euclidean','l1','l2','manhattan','braycurtis','canberra','chebyshev','correlation','dice','hamming','jaccard','kulsinski','mahalanobis','matching','minkowski','rogerstanimoto','russellrao','seuclidean','sokalmichener','sokalsneath','sqeuclidean','yule']), show_default=True)
@click.option('-nn', '--n_neighbors', default=10, help='Number of nearest neighbors in generation of nearest neighbor graph.', show_default=True)
@click.option('-wnn', '--weighted_nn', default=0, help='Whether to weight spectral graph for spectral clustering.', show_default=True)
@click.option('-all', '--grab_all', default=0, help='Whether to grab all clusters, and number of clusters would equal number of subgenomes. By default, the number of clusters is one greater than the number of subgenomes, and the algorithm attempts to get rid of the ambiguous cluster', show_default=True)
@click.option('-g','--gamma', default = 1.0, show_default=True, help='Gamma hyperparameter for Spectral Clustering and BGMM models.')
@click.option('-mst', '--min_span_tree', default = 0 , show_default=True, help='Augment k nearest neighbors graph with minimum spanning tree.')
def cluster(file, cluster_method, n_subgenomes, metric, n_neighbors, weighted_nn, grab_all, gamma, min_span_tree):
    """Perform clustering on the dimensionality reduced data, though various methods.
    Takes in npy position file and outputs clustered regions and plot.
    """
    if grab_all:
        n_clusters = n_subgenomes
    else:
        n_clusters = n_subgenomes + 1
    clustering_algorithms = {'SpectralClustering': SpectralClustering(n_clusters=n_clusters, eigen_solver='amg', affinity= 'precomputed', random_state=RANDOM_STATE),
                            'hdbscan_genetic':GeneticHdbscan(metric=metric,min_clusters=n_clusters,max_cluster_size=100,max_samples=50,validity_measure='silhouette', generations_number=10, gene_mutation_prob=0.45, gene_crossover_prob = 0.45, population_size=100 , interval=100, upper_bound = True),
                             'KMeans': MiniBatchKMeans(n_clusters=n_clusters),
                             'GMM':GaussianMixture(n_components=n_clusters),
                             'BGMM':BayesianGaussianMixture(n_components=n_clusters, weight_concentration_prior_type=('dirichlet_distribution' if weighted_nn else 'dirichlet_process'), weight_concentration_prior=(gamma if gamma != 1.0 else None))}
    n_clusters = n_subgenomes + 1
    name, algorithm = cluster_method , clustering_algorithms[cluster_method]
    dataOld = sps.load_npz('clusteringMatrix.npz')
    scaffolds = pickle.load(open('scaffolds.p', 'rb'))
    Tname = file.split('transformed3D')[0]
    transformed_data = np.load(file)
    n_dimensions = transformed_data.shape[1]
    transformed_data = StandardScaler().fit_transform(transformed_data)
    if os.path.exists(name + Tname + 'n%d' % n_clusters + 'ClusterTest.html') == 0:
        try:
            os.makedirs('analysisOutputs/' + name + Tname + 'n%d' % n_clusters)
        except:
            pass
        try:
            os.makedirs('analysisOutputs/' + name + Tname + 'n%d' % n_clusters+'/clusterResults')
        except:
            pass
        if cluster_method == 'SpectralClustering':
            neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm = 'brute', metric=metric)
            neigh.fit(transformed_data)
            fit_data = neigh.kneighbors_graph(transformed_data, mode = ('connectivity' if weighted_nn == 0 else 'distance'))
            if min_span_tree == 0:
                connected = sps.csgraph.connected_components(fit_data)
                if connected[0] > 1:
                    counts = Counter(connected[1])
                    subgraph_idx = max(counts.iteritems(), key=lambda x: x[1])[0]
                    scaffBool = connected[1] == subgraph_idx
                    dataOld = dataOld[scaffBool]
                    transformed_data = transformed_data[scaffBool,:]
                    scaffolds_noconnect = list(np.array(scaffolds)[scaffBool == False])
                    scaffolds = list(np.array(scaffolds)[scaffBool])
                    n_connected = connected[0]
                    while(n_connected > 1):
                        neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric=metric)
                        neigh.fit(transformed_data)
                        fit_data = neigh.kneighbors_graph(transformed_data, mode = ('connectivity' if weighted_nn == 0 else 'distance'))
                        connected = sps.csgraph.connected_components(fit_data)
                        counts = Counter(connected[1])
                        subgraph_idx = max(counts.iteritems(), key=lambda x: x[1])[0]
                        scaffBool = connected[1] == subgraph_idx
                        if connected[0] > 1:
                            dataOld = dataOld[scaffBool]
                            transformed_data = transformed_data[scaffBool, :]
                            scaffolds_noconnect += list(np.array(scaffolds)[scaffBool == False])
                            scaffolds = list(np.array(scaffolds)[scaffBool])
                            #print 'c',len(scaffolds), transformed_data.shape
                        n_connected = connected[0]
                else:
                    scaffolds_noconnect = []
            else:
                mst = sps.csgraph.minimum_spanning_tree(fit_data).tocsc() # fixme instead find MST of fully connected graph fitdata # pairwise_distances(transformed_data,metric=metric)
                #min_span_tree = (min_span_tree + min_span_tree.T > 0).astype(int)
                fit_data += mst
                fit_data += fit_data.T
                fit_data = (fit_data > 0).astype(np.float)
                #print fit_data.todense()
                #print sps.csgraph.connected_components(fit_data)

                # FIXME union between this and nearest neighbors check for memory issues, and change to transformed data
            if n_dimensions > 3:
                t_data = KernelPCA(n_components=3, random_state=RANDOM_STATE).fit_transform(transformed_data)
            else:
                t_data = transformed_data
            np.save('analysisOutputs/' + name + Tname + 'n%d' % n_clusters +'/graphInitialPositions.npy', t_data)
            del t_data
            sps.save_npz('analysisOutputs/' + name + Tname + 'n%d' % n_clusters +'/spectralGraph.npz', fit_data.tocsc())
            pickle.dump(scaffolds,open('analysisOutputs/' + name + Tname + 'n%d' % n_clusters + '/scaffolds_connect.p', 'wb'))
            if not min_span_tree:
                pickle.dump(scaffolds_noconnect,open('analysisOutputs/' + name + Tname + 'n%d' % n_clusters + '/scaffolds_noconnect.p', 'wb'))
        else:
            fit_data = pairwise_distances(transformed_data,metric=metric) if name == 'hdbscan_genetic' and metric in ['ectd','cosine','mahalanobis'] else transformed_data
        algorithm.fit(fit_data)
        if n_dimensions > 3:
            reduction = KernelPCA(n_components=3, random_state=RANDOM_STATE)
            reduction.fit(transformed_data)
            reductionT = reduction.transform(transformed_data)
            scaledfit = StandardScaler()
            scaledfit.fit(reductionT)
            transformed_data2 = scaledfit.transform(reductionT)
        else:
            transformed_data2 = transformed_data
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(transformed_data)
        N = len(set(y_pred))
        c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N)]
        plots = []
        clusterSize = defaultdict(list)
        #print y_pred
        for key in set(y_pred):
            cluster_scaffolds = np.array(scaffolds)[y_pred == key]
            #print key, y_pred[0:10], cluster_scaffolds[0:10], scaffolds[0:10]
            if list(cluster_scaffolds):
                clusterSize[key] = np.mean(np.apply_along_axis(lambda x: np.linalg.norm(x),1,transformed_data[y_pred == key,:]))
                if clusterSize[key] == min(clusterSize.values()):
                    testCluster = key
                plots.append(
                    go.Scatter3d(x=transformed_data2[y_pred == key, 0], y=transformed_data2[y_pred == key, 1],
                                 z=transformed_data2[y_pred == key, 2],
                                 name='Cluster %d, %d points, %f distance' % (key, len(cluster_scaffolds),clusterSize[key]), mode='markers',
                                 marker=dict(color=c[key], size=2), text=cluster_scaffolds))
            with open('analysisOutputs/' + name + Tname + 'n%d' % n_clusters + '/clusterResults/subgenome_%d.txt' % key, 'w') as f:
                f.write('\n'.join(np.array(scaffolds)[y_pred == key]))
        if hasattr(algorithm, 'cluster_centers_'):
            if n_dimensions > 3:
                centers = scaledfit.transform(reduction.transform(algorithm.cluster_centers_))
            else:
                centers = algorithm.cluster_centers_
            plots.append(
                go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode='markers',
                             marker=dict(color='purple', symbol='circle', size=12),
                             opacity=0.4,
                             name='Centroids'))
        try:
            os.makedirs('analysisOutputs/' + name + Tname + 'n%d' % n_clusters + '/bootstrap_0')
        except:
            pass
        if grab_all:
            for key in set(y_pred):
                with open('analysisOutputs/' + name + Tname + 'n%d' % n_clusters + '/bootstrap_0/subgenome_%d.txt' % key, 'w') as f:
                    f.write('\n'.join(np.array(scaffolds)[y_pred == key]))
        else:
            for key in set(y_pred)-{testCluster if name != 'hdbscan_genetic' else -1}:
                with open('analysisOutputs/' + name + Tname + 'n%d' % n_clusters + '/bootstrap_0/subgenome_%d.txt' % key, 'w') as f:
                    f.write('\n'.join(np.array(scaffolds)[y_pred == key]))
        fig = go.Figure(data=plots)
        subprocess.call('touch ' + 'analysisOutputs/' + name + Tname + 'n%d' % n_clusters + '.txt',
                        shell=True)
        py.plot(fig, filename=name + Tname + 'n%d' % n_clusters + 'ClusterTest.html', auto_open=False)
        return y_pred


@polycracker.command(name='reset_cluster')
def reset_cluster():
    """Delete cluster results, subgenome extraction results and corresponding html files."""
    subprocess.call('rm analysisOutputs/* *Test.html nohup.out -r',shell=True)


@polycracker.command(name='spectral_embed_plot')
@click.option('-npy', '--positions', help='Transformed data input, npy file.',type=click.Path(exists=False))
@click.pass_context
def spectral_embed_plot(ctx, positions):
    """Spectrally embed PCA data of any origin."""
    spectral = SpectralEmbedding(n_components=3, random_state=RANDOM_STATE)
    np.save('spectral_embed.npy',spectral.fit_transform(np.load(positions)))
    spectral = SpectralEmbedding(n_components=2, random_state=RANDOM_STATE)
    t_data = spectral.fit_transform(np.load(positions))
    plt.figure()
    plt.scatter(t_data[:,0],t_data[:,1],)
    plt.savefig('spectral_embed.png')
    ctx.invoke(plotPositions, positions_npy='spectral_embed.npy',colors_pickle='xxx',output_fname='spectral_embed.html')


# begining of subgenome extraction
def fai2bed(genome):
    """Convert polyploid fai to bed file"""
    Fasta(genome)
    bedFastaDict = defaultdict(list)
    with open(genome+'.fai','r') as f, open(genome + '.bed','w') as f2:
        for line in f:
            if line:
                lineList = line.split('\t')
                bedline = '%s\t%s\t%s\t%s\n'%(lineList[0],'0',lineList[1],lineList[0])
                f2.write(bedline)
                bedFastaDict[lineList[0]] = [bedline]
    return bedFastaDict


def writeKmerCountSubgenome(subgenomeFolder, kmer_length, blast_mem, diff_kmer_threshold, diff_sample_rate, default_kmercount_value):
    """Find kmer counts in each subgenome to find differential kmers in each subgenome and extract them"""
    blast_memStr = "export _JAVA_OPTIONS='-Xms5G -Xmx%sG'" % (blast_mem)
    kmer_lengths = kmer_length.split(',')
    try:
        os.makedirs(subgenomeFolder+'/kmercount_files/')
    except:
        pass
    kmercount_path = subgenomeFolder+'/kmercount_files/'
    for fastaFile in os.listdir(subgenomeFolder):
        if 'higher.kmers' not in fastaFile and '_split' not in fastaFile and (fastaFile.endswith('.fa') or fastaFile.endswith('.fasta')):
            kmerFiles = []
            f = fastaFile.rstrip()
            outFileNameFinal = f[:f.rfind('.')] + '.kcount'
            open(kmercount_path + '/' + outFileNameFinal, 'w').close()
            for kmerL in kmer_lengths:
                print f
                outFileName = f[:f.rfind('.')] + kmerL + '.kcount'
                kmerFiles.append(kmercount_path + '/' + outFileName)
                lineOutputList = [subgenomeFolder+'/', fastaFile, kmercount_path, outFileName,kmerL]
                if int(kmerL) <= 31:
                    subprocess.call(blast_memStr + ' && module load bbtools && kmercountexact.sh overwrite=true fastadump=f mincount=3 in=%s/%s out=%s/%s k=%s -Xmx%sg' % tuple(
                            lineOutputList+[blast_mem]),shell=True)
                else:
                    subprocess.call('jellyfish count -m %s -s %d -t 15 -C -o %s/mer_counts.jf %s/%s && jellyfish dump %s/mer_counts.jf -c > %s/%s'%(kmerL,os.stat(subgenomeFolder+'/'+fastaFile).st_size,kmercount_path,subgenomeFolder+'/',fastaFile,kmercount_path,kmercount_path,outFileName),shell=True)
            subprocess.call('cat %s > %s'%(' '.join(kmerFiles), kmercount_path + '/' + outFileNameFinal), shell=True)
            subprocess.call('rm %s'%(' '.join(kmerFiles)), shell=True)
    compareKmers(diff_kmer_threshold, kmercount_path,diff_sample_rate, default_kmercount_value)


def kmercounttodict(kmercount2fname,kmercount_path):
    """kmercounttodict function creates kmer : count key value pairs, takes path and file name of a kmer count file"""
    inputFile = open(kmercount_path+kmercount2fname,'r')
    print 'my input to kcount to dict is: %s' % inputFile
    dictConverted = {}
    for line in inputFile:
        if line and len(line.split()) == 2:
            lineList = line.split()
            dictConverted[lineList[0]]=(int(lineList[1].strip('\n')))
    inputFile.close()
    return dictConverted


def compareKmers(diff_kmer_threshold,kmercount_path,diff_sample_rate,default_kmercount_value):
    """Find differential kmers between subgenomes via found kmer counts and extract differential kmers"""
    ratio_threshold = diff_kmer_threshold
    if diff_sample_rate < 1:
        diff_sample_rate = 1
    dictOfGenes = {}
    kmercountFiles = [file for file in os.listdir(kmercount_path) if file.endswith('.kcount') and '_split' not in file]
    for file in kmercountFiles:
        # creates a dictionary that associates a species to its dictionary of the kmer : count key value pairs
        # kmercounttodict function is called to create the kmer : count key value pairs
        dictOfGenes[file[:file.rfind('.')]] = kmercounttodict(file,kmercount_path)
    # output kmers and counts for differential kmers
    # output file names
    outFileNames = defaultdict(list)
    for file in kmercountFiles:
        outFileNames[kmercount_path + "/%s.higher.kmers.fa" % (file.split('.')[0])] = dictOfGenes[file[:file.rfind('.')]]
    # {file.higherkmer : {kmer:count}}
    # create files for writing
    for filename in outFileNames:
        open(filename, 'w').close()
        print 'creating %s' % filename
    for outfilename, dict1 in outFileNames.iteritems():
        # check dict 1 against other dictionaries
        out1 = open(outfilename, 'w')
        # iterate through the keys of dict1 and identify kmers that are at least ratio_threshold fold higher in dict1
        if diff_sample_rate > 1:
            count = 0
            for key, value in dict1.iteritems():
                val1 = value
                values = []
                for outfilename2 in outFileNames:
                    if outfilename2 != outfilename:
                        values.append(outFileNames[outfilename2].get(key,default_kmercount_value))
                # require at least ratio_threshold fold higher kmers in dict1 fixme ADD chi-squared test to see if it differs from genome length differences between subgenomes
                if any([(val1 / val2) > ratio_threshold for val2 in values]):
                    if count % diff_sample_rate == 0:
                        out1.write('>%s.%d.%s\n%s\n' % (key, val1, '.'.join(map(str,values)), key))
                    count += 1
        else:
            for key, value in dict1.iteritems():
                val1 = value
                values = []
                for outfilename2 in outFileNames:
                    if outfilename2 != outfilename:
                        values.append(outFileNames[outfilename2].get(key,default_kmercount_value))
                # require at least ratio_threshold fold higher kmers in dict1
                if any([(val1 / val2) > ratio_threshold for val2 in values]):
                    out1.write('>%s.%d.%s\n%s\n' % (key, val1, '.'.join(map(lambda x: str(int(x)),values)), key))
        out1.close()


def writeBlast(genome, blastPath, kmercount_path, fasta_path, bb, blast_mem, search_length = 13, perfect_mode = 1):
    """Make blast database for whole genome assembly, blast differential kmers against subgenomes"""
    create_path(blastPath)
    genome_name = genome[:genome.rfind('.')]
    blast_memStr = "export _JAVA_OPTIONS='-Xms5G -Xmx%sG'" % (blast_mem)
    for file in [file2 for file2 in os.listdir(kmercount_path) if 'higher.kmers' in file2 and (file2.endswith('.fa') or file2.endswith('.fasta'))]:
        inputFile = kmercount_path+file
        f = file.rstrip()
        outFileName = f[:f.rfind('.')]+'.BLASTtsv.txt'
        lineOutputList = [genome_name, inputFile, blastPath, outFileName]
        if bb:
            subprocess.call(blast_memStr + ' && bbmap.sh vslow=t ambiguous=all noheader=t secondary=t threads=4 maxsites=2000000000 k=%d perfectmode=%s outputunmapped=f ref=%s in=%s path=%s/ outm=%s'%(search_length,'t' if perfect_mode else 'f',fasta_path+genome,inputFile,blastPath,blastPath+'/'+f[:f.rfind('.')]+'.sam'),shell=True)
        else:
            subprocess.call(blast_memStr + ' && module load blast+/2.6.0 && blastn -db ./%s.blast_db -query %s -task "blastn-short" -outfmt 6 -out %s/%s -num_threads 4 -evalue 1e-2' % tuple(lineOutputList),shell=True)


def blast2bed3(subgenomeFolder,blastPath, bedPath, sortPath, genome,bb, bedgraph=1, out_kmers = 0):
    """Takes a list of genotype files with only one column for pos and converts them to proper bedgraph format to be sorted"""
    create_path(bedPath)
    print 'blast files contains'
    blast_files = os.listdir(blastPath)
    print('\n'.join('{}: {}'.format(*k) for k in enumerate(blast_files)))
    if bb:
        endname = '.sam'
    else:
        endname = '.BLAST.tsv.txt'
    for blast_file in blast_files:
        if blast_file.endswith(endname):
            f = blast_file.rstrip()
            outFileName = f[:f.rfind('.')]+'.bed3'
            input_list = [blastPath, f]
            inpath = '%s/%s' % tuple(input_list)
            inputFile = open(inpath, 'r')
            outpath = os.path.join(bedPath, outFileName)
            bo = open(outpath, 'w')
            if bb:
                if out_kmers:
                    for line in inputFile:
                        lineInList = line.split()
                        lineOutputList = [lineInList[2],int(lineInList[3]),int(lineInList[3])+1,lineInList[0].split('.')[0]]
                        bo.write('%s\t%d\t%d\t%s\n' % tuple(lineOutputList))
                else:
                    for line in inputFile:
                        lineInList = line.split()
                        lineOutputList = [lineInList[2],int(lineInList[3]),int(lineInList[3])+1]
                        bo.write('%s\t%d\t%d\n' % tuple(lineOutputList))
                        """blast:ATATGTTGTAATATTTGAGCACT.322.13	Nt01_118425000_118500000	100.000	23	23	24631	24609	2.35e-04	46.1"""
                        """sam:ATATGTTGTAATATTTGAGCACT.322.13  16      Nt01_118425000_118500000        24609   3       23=     *       0       0       AGTGCTCAAATATTACAACATAT *       XT:A:R  NM:i:0  AM:i:3"""
            else:
                for line in inputFile:
                    lineInList = line.split()
                    lineOutputList = [lineInList[1], int(lineInList[8]), int(lineInList[8])+1]
                    bo.write('%s\t%d\t%d\n' % tuple(lineOutputList))
            inputFile.close()
            bo.close()
            if bedgraph:
                sortedName = f[:f.rfind('.')] + '.sorted.bed3'
                si = os.path.join(bedPath, outFileName)
                so = os.path.join(sortPath, sortedName)
                coveragename = subgenomeFolder + '/' + f[:f.rfind('.')] + '.sorted.cov'
                if not os.path.exists(sortPath):
                    os.makedirs(sortPath)
                b = BedTool(si)
                if not os.path.exists(genome.replace('.fai','')+'.bed'):
                    fai2bed(genome)
                shutil.copy(genome.replace('.fai','')+'.bed',subgenomeFolder)
                windows = '%s.bed' % genome
                a = BedTool(windows)
                b.sort().saveas(so)
                a.coverage(b).saveas(coveragename)
                bedgname = f[:f.rfind('.')] + '.sorted.cov.bedgraph'
                open(subgenomeFolder + '/' + bedgname, 'w').close()
                bedgo = open(subgenomeFolder + '/' + bedgname, 'w')
                covFile = open(coveragename, 'r')
                for line in covFile:
                    lineInList = line.split()
                    lineOutputList = [lineInList[0], int(lineInList[1]), int(lineInList[2]), int(lineInList[5]) ]
                    bedgo.write('%s\t%d\t%d\t%d\n' % tuple(lineOutputList))
                covFile.close()
                bedgo.close()


def bed2unionBed(genome, subgenomeFolder, bedPath):
    """Convert bedgraph files into union bedgraph files"""
    bedGraphFiles = [file for file in os.listdir(subgenomeFolder) if file.endswith('.sorted.cov.bedgraph')]
    inputName = 'subgenomes'
    outputFileName = subgenomeFolder + '/' +inputName + '.union.bedgraph'
    genome_name = genome[:genome.rfind('.')]
    subprocess.call('cut -f 1-2 %s.fai > %s.genome'%(genome,subgenomeFolder + '/genome'),shell=True)
    genomeFile = subgenomeFolder + '/genome.genome'
    bedGraphBedSortFn = [BedTool(subgenomeFolder+'/' + file).sort().fn for file in bedGraphFiles]
    x = BedTool()
    result = x.union_bedgraphs(i=bedGraphBedSortFn, g=genomeFile, empty=True)
    result.saveas(outputFileName)


@polycracker.command(name='kmerratio2scaffasta')
@click.pass_context
def kmerratio2scaffasta(ctx,subgenome_folder, original_subgenome_path, fasta_path, genome_name, original_genome, bb, bootstrap, iteration, kmer_length, run_final, original, blast_mem, kmer_low_count, diff_kmer_threshold, unionbed_threshold, diff_sample_rate, default_kmercount_value, search_length, perfect_mode):
    """Bin genome regions into corresponding subgenomes based on the kmer counts in each region and their corresponding subgenomes"""
    try:
        absolute_threshold, ratio_threshold = tuple(map(int,unionbed_threshold.split(',')))
    except:
        absolute_threshold, ratio_threshold = 10, 2
    if original:
        genome_name = original_genome
    genome = fasta_path + genome_name
    blast_memStr = "export _JAVA_OPTIONS='-Xms5G -Xmx%sG'" % (blast_mem)
    a = subgenome_folder + '/subgenomes.union.bedgraph'
    ubedg = open(a, 'r')
    genomeFastaObj = Fasta(genome)
    extractPath = subgenome_folder+'/extractedSubgenomes/'
    try:
        os.makedirs(extractPath)
    except:
        pass
    genomeprefix = genome[genome.rfind('/')+1:genome.rfind('.')]
    outputSubgenomes = [extractPath + genomeprefix + '.subgenome' + chr(i+65) + '.fasta' for i in range(len(ubedg.readline().split('\t')[3:]))]
    ubedg.seek(0)
    scaffoldsOut = [[] for subgenome in outputSubgenomes]
    ambiguousScaffolds = []
    # parse the unionbed file to subset
    for line in ubedg:
        if line:
            lineList = line.split('\t')
            scaff = str((lineList[0]).rstrip())
            ambiguous = 1
            x = [float((lineList[i]).rstrip()) for i in range(3,len(lineList))]
            for i in range(len(x)):
                x_others = x[:i] + x[i+1:]
                if all([(x_i == 0 and x[i] > absolute_threshold) or (x_i > 0 and (x[i]/x_i) > ratio_threshold) for x_i in x_others]):
                    scaffoldsOut[i].append(scaff)
                    ambiguous = 0
            if ambiguous:
                ambiguousScaffolds.append(scaff)
    no_kill = all([len(subgenome) > 0 for subgenome in scaffoldsOut])
    ubedg.close()
    for subgenome, scaffolds in zip(outputSubgenomes,scaffoldsOut):
        with open(subgenome,'w') as f:
            for scaff in scaffolds:
                f.write('>%s\n%s\n' % (scaff, str(genomeFastaObj[scaff][:])))
        subprocess.call(blast_memStr + ' && reformat.sh in=%s out=%s fastawrap=60'%(subgenome,subgenome.replace('.fasta','_wrapped.fasta')),shell=True)
    with open(extractPath+'ambiguousScaffolds.fasta','w') as f:
        for scaff in ambiguousScaffolds:
            f.write('>%s\n%s\n' % (scaff, str(genomeFastaObj[scaff][:])))
    subprocess.call(blast_memStr + ' && reformat.sh in=%s out=%s fastawrap=60' % (extractPath+'ambiguousScaffolds.fasta', extractPath+'ambiguousScaffolds_wrapped.fasta'), shell=True)
    if subgenome_folder.endswith('/'):
        subgenome_folder = subgenome_folder[:subgenome_folder.rfind('/')]
    if iteration < bootstrap:
        print 'NOT FINAL ITERATION'
        subgenome_folder = subgenome_folder[:subgenome_folder.rfind('/')] + '/bootstrap_%d' % (iteration + 1)
        try:
            os.makedirs(subgenome_folder)
        except:
            pass
        for i, scaffolds in enumerate(scaffoldsOut):
            if scaffolds:
                with open(subgenome_folder + '/subgenome_%d.txt'%i,'w') as f:
                    f.write('\n'.join(scaffolds))
    elif iteration == bootstrap:
        print 'FINAL ITERATION'
        iteration += 1
        try:
            os.makedirs(original_subgenome_path + '/finalResults')
        except:
            pass
        for i, scaffolds in enumerate(scaffoldsOut):
            if scaffolds:
                with open(original_subgenome_path + '/finalResults/subgenome_%d.txt' % i, 'w') as f:
                    f.write('\n'.join(scaffolds))
        run_final = 1
    if no_kill == 0:
        print 'DEAD'
        return
    if run_final == 0:
        iteration += 1
        print 'ITERATION'
        ctx.invoke(subgenomeExtraction, subgenome_folder=subgenome_folder, original_subgenome_path=original_subgenome_path, fasta_path=fasta_path, genome_name=genome_name, original_genome=original_genome, bb=bb, bootstrap=bootstrap, iteration=iteration, kmer_length=kmer_length, run_final=run_final, original=original, blast_mem=blast_mem, kmer_low_count=kmer_low_count, diff_kmer_threshold=diff_kmer_threshold, unionbed_threshold=unionbed_threshold, diff_sample_rate=diff_sample_rate, default_kmercount_value=default_kmercount_value, search_length = search_length, perfect_mode = perfect_mode)
    else:
        print 'DONE'
        return
# FIXME add hyperparameter scan using GA here!


@polycracker.command(name='subgenomeExtraction')
@click.option('-s', '--subgenome_folder', help='Subgenome folder where subgenome extraction is taking place. This should be the bootstrap_0 folder.', type=click.Path(exists=False))
@click.option('-os', '--original_subgenome_path', help='One level up from subgenomeFolder. Final outputs are stored here.', type=click.Path(exists=False))
@click.option('-p', '--fasta_path', default='./fasta_files/', help='Directory containing chunked polyploids fasta file', show_default=True, type=click.Path(exists=False))
@click.option('-g', '--genome_name', help='Filename of chunked polyploid fasta.', type=click.Path(exists=False))
@click.option('-go', '--original_genome', help='Filename of polyploid fasta; can be either original or chunked, just make sure original is set to 1 if using original genome.', type=click.Path(exists=False))
@click.option('-bb', '--bb', default=1, help='Whether bbtools were used in generating blasted sam file.', show_default=True)
@click.option('-b', '--bootstrap', default=0, help='Number of times to bootstrap the subgenome extraction process. Each time you bootstrap, you should get better classification results, else instability has occurred due to ambiguous clustering.', show_default=True)
@click.option('-i', '--iteration', default=0, help='Current iteration of bootstrapping process. Please set to 0, when initially beginning.', show_default=True)
@click.option('-l', '--kmer_length', default='23,31', help='Length of kmers to find; can include multiple lengths if comma delimited (e.g. 23,25,27)', show_default=True)
@click.option('-r', '--run_final', default=0, help='Turn this to 0 when starting the command. Script terminates if this is set to 1.', show_default=True)
@click.option('-o', '--original', default=0, help='Select 1 if trying to extract subgenomes based on original nonchunked genome.', show_default=True)
@click.option('-m', '--blast_mem', default='100', help='Amount of memory to use for bbtools run. More memory will speed up processing.', show_default=True)
@click.option('-kl', '--kmer_low_count', default=100, help='Omit kmers from analysis that have less than x occurrences throughout genome.', show_default=True)
@click.option('-dk', '--diff_kmer_threshold', default=20, help='Value indicates that if the counts of a particular kmer in a subgenome divided by any of the counts in other subgenomes are greater than this value, then the kmer is differential.', show_default=True)
@click.option('-u', '--unionbed_threshold', default='10,2', help='Two comma delimited values used for binning regions into subgenomes. First value indicates that if the counts of all kmers in a particular region in any other subgenomes are 0, then the total count in a subgenome must be more than this much be binned as that subgenome. Second value indicates that if any of the other subgenome kmer counts for all kmers in a region are greater than 0, then the counts on this subgenome divided by any of the other counts must surpass a value to be binned as that subgenome.', show_default=True)
@click.option('-ds', '--diff_sample_rate', default=1, help='If this option, x, is set greater than one, differential kmers are sampled at lower frequency, and total number of differential kmers included in the analysis is reduced to (total #)/(diff_sample_rate), after threshold filtering', show_default=True)
@click.option('-kv', '--default_kmercount_value', default = 3., help='If a particular kmer is not found in a kmercount dictionary, this number, x, will be used in its place. Useful for calculating differential kmers.', show_default = True)
@click.option('-sl', '--search_length', default=13, help='Kmer length for mapping.', show_default=True)
@click.option('-pm', '--perfect_mode', default = 1, help='Perfect mode.', show_default=True)
@click.pass_context
def subgenomeExtraction(ctx, subgenome_folder, original_subgenome_path, fasta_path, genome_name, original_genome, bb, bootstrap, iteration, kmer_length, run_final, original, blast_mem, kmer_low_count, diff_kmer_threshold, unionbed_threshold, diff_sample_rate, default_kmercount_value, search_length, perfect_mode):
    """Extract subgenomes from genome, either chunked genome or original genome can be fed in as argument"""
    if not fasta_path.endswith('/'):
        fasta_path += '/'
    bedDict = fai2bed(fasta_path+genome_name)
    blastPath = subgenome_folder + '/blast_files/'
    bedPath = subgenome_folder + '/bed_files/'
    sortPath = subgenome_folder + '/sortedbed_files/'
    create_path(blastPath)
    create_path(bedPath)
    create_path(sortPath)
    if bootstrap >= iteration or run_final == 1:
        for file in os.listdir(subgenome_folder):
            if file and file.endswith('.txt') and os.stat(subgenome_folder+'/'+file).st_size:
                with open(subgenome_folder+'/'+file,'r') as f,open(subgenome_folder+'/'+file.replace('.txt','.bed'),'w') as f2:
                    for line in f:
                        if line:
                            f2.write(bedDict[line.strip('\n')][0])
                subprocess.call('bedtools getfasta -fi %s -fo %s -bed %s -name'%(fasta_path+genome_name,subgenome_folder + '/%s_'%('model')+file.replace('.txt','.fa'),subgenome_folder+'/'+file.replace('.txt','.bed')),shell=True)
        writeKmerCountSubgenome(subgenome_folder,kmer_length,blast_mem,diff_kmer_threshold,diff_sample_rate, default_kmercount_value)
        writeBlast(original_genome,blastPath,subgenome_folder+'/kmercount_files/',fasta_path,bb,blast_mem, search_length, perfect_mode)
        blast2bed3(subgenome_folder, blastPath, bedPath, sortPath, fasta_path+original_genome, bb)
        bed2unionBed(fasta_path+original_genome, subgenome_folder, bedPath)
        ctx.invoke(kmerratio2scaffasta, subgenome_folder=subgenome_folder, original_subgenome_path=original_subgenome_path, fasta_path=fasta_path, genome_name=genome_name, original_genome=original_genome, bb=bb, bootstrap=bootstrap, iteration=iteration, kmer_length=kmer_length, run_final=run_final, original=original, blast_mem=blast_mem, kmer_low_count=kmer_low_count, diff_kmer_threshold=diff_kmer_threshold, unionbed_threshold=unionbed_threshold, diff_sample_rate=diff_sample_rate, default_kmercount_value=default_kmercount_value, search_length = search_length, perfect_mode = perfect_mode)
    if bootstrap < iteration and run_final == 0:
        subgenome_folder = original_subgenome_path
        subgenome_folder, run_final = 'null', 0
        iteration += 1
        if run_final:
            ctx.invoke(subgenomeExtraction, subgenome_folder=subgenome_folder, original_subgenome_path=original_subgenome_path, fasta_path=fasta_path, genome_name=genome_name, original_genome=original_genome, bb=bb, bootstrap=bootstrap, iteration=iteration, kmer_length=kmer_length, run_final=run_final, original=original, blast_mem=blast_mem, kmer_low_count=kmer_low_count, diff_kmer_threshold=diff_kmer_threshold, unionbed_threshold=unionbed_threshold, diff_sample_rate=diff_sample_rate, default_kmercount_value=default_kmercount_value, search_length = search_length, perfect_mode = perfect_mode)


@polycracker.command(name='cluster_exchange')
@click.option('-s', '--subgenome_folder', help='Subgenome folder where subgenome extraction is taking place. Parent of the bootstrap_0 folder.', type=click.Path(exists=False))
@click.option('-l', '--list_of_clusters', help='Comma delimited list of numbers corresponding to cluster names. E.g. 0,1,2 corresponds to subgenome_0,subgenome_1, and subgenome_2, and those would be extracted.', type=click.Path(exists=False))
def cluster_exchange(subgenome_folder,list_of_clusters):
    """Prior to subgenome Extraction, can choose to switch out clusters, change clusters under examination, judging from clustered PCA plot if incorrect ambiguous cluster was found."""
    print 'Make sure to delete other bootstrap folders other than bootstrap_0 and also finalResults.'
    subprocess.call('rm %s/finalResults %s %s/bootstrap_0/* -r ; scp %s %s/bootstrap_0'%(subgenome_folder,' '.join([subgenome_folder+'/'+folder for folder in os.listdir(subgenome_folder) if folder.startswith('bootstrap') and folder.endswith('0')==0]),subgenome_folder,' '.join([subgenome_folder+'/clusterResults/subgenome_%s.txt'%i for i in list_of_clusters.split(',')]),subgenome_folder),shell=True)


@polycracker.command(name='txt2fasta')
@click.option('-txt', '--txt_files', default='subgenome_1.txt,subgenome_2.txt', show_default=True, help='Comma delimited list of text files or folder containing text files.', type=click.Path(exists=False))
@click.option('-rf', '--reference_fasta', help='Full path to reference fasta file containing scaffold names.', type=click.Path(exists=False))
def txt2fasta(txt_files,reference_fasta):
    """Extract subgenome fastas from reference fasta file using polyCRACKER found text files of subgenome binned scaffolds."""
    if '.txt' in txt_files:
        txt_files = txt_files.split(',')
    else:
        txt_files = [txt_files+'/'+f for f in os.listdir(txt_files) if f.endswith('.txt')]
    for txt_file in txt_files:
        subprocess.call('export _JAVA_OPTIONS="-Xmx5g" && filterbyname.sh overwrite=t in=%s out=%s names=%s include=t'%(reference_fasta,txt_file.replace('.txt','.fa'),txt_file),shell=True)


@polycracker.command(name='clusterGraph')
@click.option('-sp', '--sparse_matrix_file', default='spectralGraph.npz', help='Sparse nearest neighbors graph npz file.', show_default=True, type=click.Path(exists=False))
@click.option('-sf', '--scaffolds_file', default='scaffolds_connect.p', help='Pickle file containing scaffold/chunk names.', show_default=True, type=click.Path(exists=False))
@click.option('-od', '--out_dir', default = './', help='Directory to output final plots.', show_default=True, type=click.Path(exists=False))
@click.option('-l', '--layout', default='standard', help='Layout from which to plot graph.', type=click.Choice(['standard','spectral','random']), show_default=True)
@click.option('-p', '--positions_npy', default='graphInitialPositions.npy', help='If standard layout, then use these data points to begin simulation.', show_default=True, type=click.Path(exists=False))
@click.option('-i', '--iteration', default='0,1,2,3', help='Can comma delimit the number of iterations you would like to simulate to. No comma for a single cycle.', show_default=True)
@click.option('-b', '--bed_features_file', default='xxx', help='Optional bed file in relation to nonchunked genome. Must include features in fourth column.', show_default=True, type=click.Path(exists=False))
def clusterGraph(sparse_matrix_file, scaffolds_file, out_dir, layout, positions_npy, iteration, bed_features_file):
    """Plots nearest neighbors graph in html format and runs a physics simulation on the graph over a number of iterations."""
    if bed_features_file.endswith('.bed') == 1:
        featureMap = 1
    else:
        featureMap = 0
    G = nx.from_scipy_sparse_matrix(sps.load_npz(sparse_matrix_file))
    scaffolds = pickle.load(open(scaffolds_file, 'rb'))
    N = 2
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N + 1)]
    mapping = {i:scaffolds[i] for i in range(len(scaffolds))}
    G=nx.relabel_nodes(G,mapping, copy=False)
    nodes = G.nodes()
    if featureMap:
        scaffoldsDict = {scaffold : '\t'.join(['_'.join(scaffold.split('_')[0:-2])]+scaffold.split('_')[-2:]) for scaffold in scaffolds}
        outputFeatures = defaultdict(list)
        scaffoldsBed = BedTool('\n'.join(scaffoldsDict.values()),from_string=True)
        featureBed = BedTool(bed_features_file)
        featuresDict = {scaffold: '' for scaffold in scaffolds}
        finalBed = scaffoldsBed.intersect(featureBed,wa=True,wb=True).sort().merge(d=-1,c=7,o='distinct')
        finalBed.saveas(out_dir+'/finalBed.bed')
        omittedRegions = scaffoldsBed.intersect(featureBed,v=True,wa=True)
        omittedRegions.saveas(out_dir+'/ommitted.bed')
        for line in str(finalBed).splitlines()+[line2+'\tunlabelled' for line2 in str(omittedRegions).splitlines()]:
            lineList = line.strip('\n').split('\t')
            feature = lineList[-1]
            scaffold = '_'.join(lineList[0:-1])
            if ',' in feature:
                featuresDict[scaffold] = '|'.join(feature.split(','))
                feature = 'ambiguous'
            else:
                featuresDict[scaffold] = feature
            outputFeatures[scaffold] = feature
        mainFeatures = set(outputFeatures.values())
        N = len(mainFeatures)
        c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N + 1)]
        featuresColors = {feature : c[i] for i, feature in enumerate(mainFeatures)}
        outputFeaturesArray = np.array([outputFeatures[scaffold] for scaffold in nodes])
        names = np.vectorize(lambda name: 'Scaffolds: ' + name)(outputFeaturesArray)
        colors = np.vectorize(lambda feature: featuresColors[feature])(outputFeaturesArray)
        nodesText = np.array(['%s, %d connections, feature= %s' % (scaffold, int(G.degree(scaffold)),featuresDict[scaffold]) for scaffold in nodes])
    else:
        names = 'Scaffolds'
        colors = c[0]
        nodesText = ['%s, %d connections, ' % (scaffold, int(G.degree(scaffold))) for scaffold in nodes]
    if layout == 'spectral':
        pos_i = nx.spectral_layout(G,dim=3)
    elif layout == 'random':
        pos_i = nx.random_layout(G, dim=3)
    else:
        pos_i = defaultdict(list)
        t_data = np.load(positions_npy)
        for i in range(len(scaffolds)):
            pos_i[scaffolds[i]] = tuple(t_data[i,:])
    try:
        iterations = sorted(map(int,iteration.split(',')))
    except:
        quit()
    masterData = []
    for idx,i in enumerate(iterations):
        if i != 0:
            pos = nx.spring_layout(G,dim=3,iterations=i,pos=pos_i)
        else:
            pos = pos_i
        plots = []
        Xv = np.array([pos[k][0] for k in nodes])
        Yv = np.array([pos[k][1] for k in nodes])
        Zv = np.array([pos[k][2] for k in nodes])
        Xed = []
        Yed = []
        Zed = []
        for edge in G.edges():
            Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
            Yed += [pos[edge[0]][1], pos[edge[1]][1], None]
            Zed += [pos[edge[0]][2], pos[edge[1]][2], None]
        if featureMap:
            for name in mainFeatures:
                plots.append(go.Scatter3d(x=Xv[outputFeaturesArray == name],
                                      y=Yv[outputFeaturesArray == name],
                                      z=Zv[outputFeaturesArray == name],
                                      mode='markers',
                                      name= name,
                                      marker=go.Marker(symbol='dot',
                                                       size=5,
                                                       color=featuresColors[name],
                                                       line=go.Line(color='rgb(50,50,50)', width=0.5)
                                                       ),
                                      text=nodesText[outputFeaturesArray == name],
                                      hoverinfo='text'
                                      ))
        else:
            plots.append(go.Scatter3d(x=Xv,
                                      y=Yv,
                                      z=Zv,
                                      mode='markers',
                                      name=names,
                                      marker=go.Marker(symbol='dot',
                                                       size=5,
                                                       color=colors,
                                                       line=go.Line(color='rgb(50,50,50)', width=0.5)
                                                       ),
                                      text=nodesText,
                                      hoverinfo='text'
                                      ))
        plots.append(go.Scatter3d(x=Xed,
                                  y=Yed,
                                  z=Zed,
                                  mode='lines',
                                  line=go.Line(color='rgb(210,210,210)', width=1),
                                  hoverinfo='none'
                                  ))
        if idx == 0:
            sliders_dict = {
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'Frame:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': []
            }
        slider_step = {'args': [
            [str(i)],
            {'frame': {'duration': 300, 'redraw': False},
             'mode': 'immediate',
             'transition': {'duration': 300}}
            ],
            'label': str(i),
            'method': 'animate'}
        sliders_dict['steps'].append(slider_step)
        masterData.append({'data' : go.Data(plots),'name' : str(i)})
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )
    masterLayout = dict(
        title="Graph of Scaffolds",
        updatemenus=[{'direction': 'left',
                      'pad': {'r': 10, 't': 87},
                      'showactive': False,
                      'type': 'buttons',
                      'x': 0.1,
                      'xanchor': 'right',
                      'y': 0,
                      'yanchor': 'top', 'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': False},
                                    'fromcurrent': True,
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                      'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ]}],
        sliders=[sliders_dict],
        width=1000,
        height=1000,
        showlegend=True,
        scene=go.Scene(
            xaxis=go.XAxis(axis),
            yaxis=go.YAxis(axis),
            zaxis=go.ZAxis(axis),
        ),
        margin=go.Margin(
            t=100
        ),
        hovermode='closest',
        annotations=go.Annotations([
            go.Annotation(
                showarrow=False,
                text="",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=go.Font(
                    size=14
                )
            )
        ]), )
    fig1 = go.Figure(data=masterData[0]['data'], layout=masterLayout, frames=masterData)
    py.plot(fig1, filename=out_dir + '/OutputGraph_frames_%s.html'%(','.join(map(str,iterations))), auto_open=False)


@polycracker.command(name='plotPositions')
@click.option('-npy', '--positions_npy', default='graphInitialPositions.npy', help='If standard layout, then use these data points to begin simulation.', show_default=True, type=click.Path(exists=False))
@click.option('-p', '--labels_pickle', default='scaffolds.p', help='Pickle file containing scaffolds.', show_default=True, type=click.Path(exists=False))
@click.option('-c', '--colors_pickle', default='colors_pickle.p', help='Pickle file containing the cluster/class each label/scaffold belongs to.', show_default=True, type=click.Path(exists=False))
@click.option('-o', '--output_fname', default = 'output.html', help='Desired output plot name in html.', show_default=True, type=click.Path(exists=False))
@click.option('-npz', '--graph_file', default='xxx', help='Sparse nearest neighbors graph npz file. If desired, try spectralGraph.npz.', type=click.Path(exists=False))
@click.option('-l', '--layout', default='standard', help='Layout from which to plot graph.', type=click.Choice(['standard','spectral','random']), show_default=True)
@click.option('-i', '--iterations', default=0, help='Number of iterations you would like to simulate to. No comma delimit, will only output a single iteration.', show_default=True)
@click.option('-s', '--graph_sampling_speed', default=1, help='When exporting the graph edges to CSV, can choose to decrease the number of edges for pdf report generation.', show_default=True)
@click.option('-ax','--axes_off', is_flag=True, help='When enabled, exports graph without the axes.')
@click.option('-cmap', '--new_colors', default='', show_default=True, help='Comma delimited list of colors if you want control over coloring scheme.')
def plotPositions(positions_npy, labels_pickle, colors_pickle, output_fname, graph_file, layout, iterations, graph_sampling_speed, axes_off, new_colors):
    """Another plotting function without emphasis on plotting the spectral graph. Emphasis is on plotting positions and clusters."""
    labels = pickle.load(open(labels_pickle,'rb'))
    if graph_file.endswith('.npz'):
        graph = 1
        G = nx.from_scipy_sparse_matrix(sps.load_npz(graph_file))
        mapping = {i:labels[i] for i in range(len(labels))}
        G=nx.relabel_nodes(G,mapping, copy=False)
        if layout == 'spectral':
            pos = nx.spring_layout(G,dim=3,iterations=iterations,pos=nx.spectral_layout(G,dim=3))
        elif layout == 'random':
            pos = nx.random_layout(G, dim=3)
        else:
            t_data = np.load(positions_npy)
            n_dimensions = t_data.shape[1]
            if n_dimensions > 3:
                t_data = KernelPCA(n_components=3).fit_transform(t_data)
            pos = nx.spring_layout(G,dim=3,iterations=iterations,pos={labels[i]: tuple(t_data[i,:]) for i in range(len(labels))})
        # NOTE that G.nodes() will cause the scaffolds to be plotted out of order, so must use labels
        transformed_data = np.array([tuple(pos[k]) for k in labels])#G.nodes()
        Xed = []
        Yed = []
        Zed = []
        for edge in G.edges():
            Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
            Yed += [pos[edge[0]][1], pos[edge[1]][1], None]
            Zed += [pos[edge[0]][2], pos[edge[1]][2], None]
        if graph_sampling_speed > 1:
            Xed2, Yed2, Zed2 = [], [], []
            for edge in G.edges()[::graph_sampling_speed]:
                Xed2 += [pos[edge[0]][0], pos[edge[1]][0], None]
                Yed2 += [pos[edge[0]][1], pos[edge[1]][1], None]
                Zed2 += [pos[edge[0]][2], pos[edge[1]][2], None]
        else:
            Xed2, Yed2, Zed2 = Xed, Yed, Zed
        pd.DataFrame(np.vstack((Xed2,Yed2,Zed2)).T,columns=['x','y','z']).to_csv(output_fname.replace('.html','_graph_connections.csv'),index=False)
        del Xed2, Yed2, Zed2
    else:
        transformed_data = np.load(positions_npy)
        n_dimensions = transformed_data.shape[1]
        if n_dimensions > 3:
            transformed_data = KernelPCA(n_components=3).fit_transform(transformed_data)
        graph = 0
    if output_fname.endswith('.html') == 0:
        output_fname += '.html'
    if colors_pickle.endswith('.p'):
        names = pickle.load(open(colors_pickle,'rb'))
        print names
        c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(set(names)) + 2)]
        if new_colors:
            c=new_colors.split(',')
        color = {name: c[i] for i,name in enumerate(sorted(set(names)))}
        print color
        df = pd.DataFrame(data={'x':transformed_data[:,0],'y':transformed_data[:,1],'z':transformed_data[:,2],'text':np.array(labels),'names':names})
        df['color'] = df['names'].map(color)# LabelEncoder().fit_transform(df['names'])
        plots = []
        """for name,col in color.items():
            print name
            plots.append(
                go.Scatter3d(x=transformed_data[names == name,0], y=transformed_data[names == name,1],
                             z=transformed_data[names == name,2],
                             name=name, mode='markers',
                             marker=dict(color=col, size=2), text=np.array(labels)[names == name]))"""
        print(color.items())
        for name,col in color.items():#enumerate(df['names'].unique()):
            plots.append(
                go.Scatter3d(x=df['x'][df['names']==name], y=df['y'][df['names']==name],
                             z=df['z'][df['names']==name],
                             name=name, mode='markers',
                             marker=dict(color=col, size=2), text=df['text'][df['names']==name]))
    else:
        N = 2
        c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N + 1)]
        plots = []
        plots.append(
            go.Scatter3d(x=transformed_data[:,0], y=transformed_data[:,1],
                         z=transformed_data[:,2],
                         name='Scaffolds', mode='markers',
                         marker=dict(color=c[0], size=2), text=labels))
    if graph:
        plots.append(go.Scatter3d(x=Xed,
                                  y=Yed,
                                  z=Zed,
                                  mode='lines',
                                  line=go.Line(color='rgb(210,210,210)', width=1),
                                  hoverinfo='none'
                                  ))
    if axes_off:
        fig = go.Figure(data=plots,layout=go.Layout(scene=dict(xaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
            yaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
            zaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False))))
    else:
        fig = go.Figure(data=plots)
    py.plot(fig, filename=output_fname, auto_open=False)
    try:
        pd.DataFrame(np.hstack((transformed_data,names[:,None])),columns=['x','y','z','Name']).to_csv(output_fname.replace('.html','.csv'))
    except:
        pass


@polycracker.command(name="number_repeatmers_per_subsequence")
@click.option('-m', '--merged_bed', default='blasted_merged.bed', help='Merged bed file containing bed subsequences and kmers, comma delimited.', show_default=True, type=click.Path(exists=False))
@click.option('-o', '--out_file', default='kmers_per_subsequence.png', help='Output histogram in png or pdf.', show_default=True, type=click.Path(exists=False))
@click.option('--kde', is_flag=True, help='Add kernel density estimation.')
def number_repeatmers_per_subsequence(merged_bed, out_file, kde):
    """Find histogram depicting number of repeat kmers per subsequence. If find tail close to zero, good choice to enfore a minimum chunk size for inclusion into kmer count matrix or change size of chunked genome fragments.
    Useful for assessing average repeat content across genome fragments. If too low, see README for recommendations."""
    plt.style.use('ggplot')
    kcount = np.genfromtxt(os.popen("awk '{print gsub(/,/,\"\")+1}' %s"%(merged_bed),'r'))
    plt.figure()
    kde_plot = sns.distplot(kcount, kde=kde)
    plt.title('Histogram of Repeat-mers in Chunked Genome Fragment')
    plt.xlabel(('Number of Repeat-mers in Chunked Genome Fragment'))
    plt.ylabel('Frequency/Density')
    if out_file.endswith('.png') == 0:
        out_file += '.png'
    plt.savefig(out_file,dpi=300)


@polycracker.command(name='kcount_hist_old')
@click.option('-k', '--kcount_file', help='Input kmer count file.', type=click.Path(exists=False))
@click.option('-o', '--out_file', default='kmer_histogram.png', help='Output histogram in png or pdf.', show_default=True, type=click.Path(exists=False))
@click.option('--log', is_flag=True, help='Scale x-axis to log base 10.')
def kcount_hist_old(kcount_file, out_file, log):
    """Outputs a histogram plot of a given kmer count file."""
    import seaborn as sns
    histValues = []
    with open(kcount_file,'r') as f:
        if log:
            for line in f:
                if line:
                    try:
                        histValues.append(np.log(int(line.strip('\n').split()[-1]))) #'\t'
                    except:
                        print line
        else:
            for line in f:
                if line:
                    try:
                        histValues.append(int(line.strip('\n').split()[-1])) #'\t'
                    except:
                        print line
    histValues = np.array(histValues)
    plt.figure()
    kde_plot = sns.distplot(histValues, kde=False)
    plt.title('KmerCount Histogram')
    plt.xlabel(('Total Kmer Counts in Genome in log(b10)x' if log else 'Total Kmer Counts in Genome'))
    plt.ylabel('Frequency/Density')
    if out_file.endswith('.png') == 0:
        out_file += '.png'
    plt.savefig(fname=out_file,dpi=300)


@polycracker.command(name='kcount_hist')
@click.option('-k', '--kcount_directory', help='Directory containing input kmer count files. Rename the kcount files to [insert-kmer-length].kcount, eg. 31.kcount . ln -s ... may be useful in this case.', type=click.Path(exists=False))
@click.option('-o', '--out_file', default='kmer_histogram.png', help='Output histogram in png.', show_default=True, type=click.Path(exists=False))
@click.option('--log', is_flag=True, help='Scale x-axis to log base 10.')
@click.option('-r','--kcount_range', default='3,10000', help='Comma delimited list of two items, specifying range of kmer counts.', show_default=True)
@click.option('-l','--low_count', default= 0., help='Input the kmer low count threshold and it will be plotted.', show_default = True)
@click.option('-kl','--kmer_lengths', default='', help='Optional: comma delimited list of kmer lengths to include in analysis. Eg. 31,54,73. Can optionally input the kmer low count corresponding to each length via another comma delimited list after colon, eg. Eg. 31,54,73:100,70,50 , where 100,70,50 are the low count thresholds')
@click.option('-s','--sample_speed', default=1, help='Optional: increase sample speed to attempt to smooth kmer count histogram. A greater sample speed will help with smoothing more but sacrifice more information.', show_default=True)
def kcount_hist(kcount_directory, out_file, log, kcount_range, low_count, kmer_lengths, sample_speed):
    """Outputs a histogram plot of a given kmer count files."""
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from scipy.signal import savgol_filter as smooth
    from scipy.interpolate import interp1d
    fig, ax = plt.subplots(figsize=(7, 7), sharex=True)
    krange = map(int,kcount_range.split(','))
    if kmer_lengths:
        kl_flag = 1
        kmer_lengths_input = kmer_lengths
        print kmer_lengths_input
        print kmer_lengths_input.split(':')[1]
        kmer_lengths = map(int,kmer_lengths.split(':')[0].split(','))
        try:
            kmer_low_counts = {kmer_lengths[i]: low_count for i,low_count in enumerate(map(int,kmer_lengths_input.split(':')[1].split(',')))}
        except:
            kmer_low_counts = 0
        print kmer_low_counts
    else:
        kl_flag = 0
        kmer_lengths = range(500)
    for file in sorted(os.listdir(kcount_directory),reverse=True):
        if file.endswith('.kcount') and (kl_flag == 0 or (kl_flag == 1 and int(file.split('.')[0]) in kmer_lengths)):
            kcount_file = kcount_directory + '/' + file
            """
            histValues = []
            with open(kcount_file,'r') as f:
                if log:
                    for line in f:
                        if line:
                            try:
                                histValues.append(np.log(int(line.strip('\n').split()[-1]))) #'\t'
                            except:
                                print line
                else:
                    for line in f:
                        if line:
                            try:
                                histValues.append(int(line.strip('\n').split()[-1])) #'\t'
                            except:
                                print line"""
            histValues = np.array(os.popen("awk '{print $2}' %s"%kcount_file).read().splitlines())
            histValues = histValues[np.where(histValues)].astype(np.int)
            if log:
                histValues = np.vectorize(lambda x: np.log(x))(histValues)
            #histValues = np.array(histValues)
            #histValues = histValues[np.where((histValues >= krange[0]) & (histValues <= krange[1]))]
            histValues = OrderedDict(sorted(Counter(histValues).items()))
            if sample_speed == 1:
                x = histValues.keys()
                y = smooth(histValues.values(),5,3)
            else:
                x = histValues.keys()[::abs(sample_speed)]
                y = smooth(histValues.values()[::abs(sample_speed)],5,3)
            f = interp1d(x,y,kind='cubic', fill_value='extrapolate')
            xnew = np.arange(krange[0],krange[1]+0.5,.01)
            plt.plot(xnew, f(xnew), label='kmer length = %s'%file.split('.')[0], hold=None)
            if kmer_low_counts:
                plt.plot(kmer_low_counts[int(file.split('.')[0])],f(kmer_low_counts[int(file.split('.')[0])]),'^',label='Kmer-Count Cutoff Value = %d'%kmer_low_counts[int(file.split('.')[0])],hold=None)
            #kde_plot = sns.distplot(histValues, hist=False, label='kmer count = %s'%file.split('.')[0], ax=ax)
    if low_count:
        plt.axvline(x=low_count,label='Kmer-Count Cutoff Value = %d'%low_count, color='xkcd:dark green')
    plt.title('KmerCount Histogram')
    plt.xlabel(('Total Kmer Counts in Genome in log(b10)x' if log else 'Total Kmer Counts in Genome'))
    plt.ylabel('Frequency')
    plt.legend()
    #if out_file.endswith('.png') == 0:
    #    out_file += '.png'
    plt.savefig(out_file,dpi=300)


@polycracker.command(name='plot_unionbed')
@click.option('-u', '--unionbed_file', help='Unionbed input file from subgenomeExtraction process or from other intermediate analyses.', type=click.Path(exists=False))
@click.option('-n', '--number_chromosomes', default=10, help='Output plots of x largest chromosomes.', show_default=True)
@click.option('-od', '--out_folder', default='./', help='Output directory for plots.', show_default=True, type=click.Path(exists=False))
def plot_unionbed(unionbed_file, number_chromosomes, out_folder): # note, cannot plot union bed file when original is set
    """Plot results of union bed file, the distribution of total differential kmer counts for each extracted subgenome as a function across the entire genome. Note that results cannot be plot if original is set to one."""
    scaffolds = defaultdict(list)
    scaffolds_size = defaultdict(list)
    scaffold_info = []
    with open(unionbed_file,'r') as f:
        for line in f:
            lineList = line.split()
            lineList2 = lineList[0].split('_')
            scaffold = '_'.join(lineList2[0:-2])
            plot_values = [np.mean(map(int,lineList2[-2:]))]+map(int,lineList[3:])
            scaffolds[scaffold].append(tuple(plot_values))
            scaffold_info.append([scaffold]+map(int,lineList2[-2:])+map(int,lineList[3:]))
    for scaffold in scaffolds:
        scaffolds[scaffold] = np.array(scaffolds[scaffold])
        scaffolds[scaffold] = scaffolds[scaffold][scaffolds[scaffold][:,0].argsort()]
        scaffolds_size[int(np.max(scaffolds[scaffold][:,0]))] = scaffold
    scaffold_info = pd.DataFrame(scaffold_info,columns=(['chr','start','end']+['subgenome%d'%i for i in range(1,np.shape(scaffolds[scaffolds.keys()[0]])[1])]))
    scaffold_info.to_csv(unionbed_file.split('/')[-1].replace('.bedgraph','.diffkmer.csv'),index=False)
    for sc in sorted(scaffolds_size.keys())[::-1][0:number_chromosomes+1]:
        scaffold = scaffolds_size[sc]
        x = scaffolds[scaffold][:,0]
        plt.figure()
        for i in range(1,np.shape(scaffolds[scaffold])[1]):
            plt.plot(x,scaffolds[scaffold][:,i],label='Subgenome %d differential kmer totals'%i)
        plt.legend()
        plt.title(scaffold)
        plt.savefig(out_folder+'/'+scaffold+'.png')
    # FIXME add function to plot bootstraps change over time, from one bootstrap to next label propagation

######################################################################################################

###################################### CIRCOS PLOTTING ANALYSES ######################################


@polycracker.command(name='generate_karyotype')
@click.option('-go', '--original_genome', default='', show_default=True, help='Filename of polyploid fasta; must be full path to original.', type=click.Path(exists=False))
@click.option('--shiny', is_flag=True, help='Export to ShinyCircos.')
def generate_karyotype(original_genome, shiny):
    """Generate karyotype shinyCircos/omicCircos csv file."""
    subprocess.call('samtools faidx '+original_genome,shell = True)
    genome_info = []
    with open(original_genome+'.fai') as f:
        if shiny:
            for line in f:
                LL = line.split()[0:2]
                genome_info.append([LL[0],1,LL[1]])
        else:
            for line in f:
                LL = line.split()[0:2]
                genome_info.append([LL[0],0,LL[1]])
    genome_info = pd.DataFrame(np.array(genome_info),columns=['chrom','chromStart','chromEnd'])
    genome_info.to_csv('genome.csv',index=False)


@polycracker.command(name='shiny2omic')
@click.option('-csv', '--input_csv', default='', show_default=True, help='Filename of shinyCircos csv.', type=click.Path(exists=False))
def shiny2omic(input_csv):
    """Convert shinyCircos csv input files to omicCircos csv input files."""
    df = pd.read_csv(input_csv)
    df['start'] = (df['start']+df['end'])/2
    df.drop(columns=['end'])
    df.rename(columns = {'start':'pos'}, inplace = True)
    df.to_csv(input_csv.replace('.csv','.omicCircos.csv'),index=False)


@polycracker.command(name='out_bed_to_circos_csv')
@click.option('-b', '--output_bed', default = 'subgenomes.bed', show_default=-True, help = 'Output bed to convert into csv file for OmnicCircos plotting.', type=click.Path(exists=False))
@click.option('-fai', '--fai_file', help='Full path to original, nonchunked fai file.', type=click.Path(exists=False))
def out_bed_to_circos_csv(output_bed,fai_file):
    """Take progenitor mapped, species ground truth, or polyCRACKER labelled scaffolds in bed file and convert for shinyCircos input of classification tracks.."""
    outname = output_bed.replace('.bed','.csv')
    final_info = []
    with open(fai_file,'r') as f:
        faiBed = BedTool('\n'.join(['\t'.join([line.split()[0]]+['0',line.split()[1]]) for line in f.read().splitlines()]),from_string=True)
    ambiguous_regions = faiBed.subtract(BedTool(output_bed))
    for line in str(ambiguous_regions).splitlines():
        if line:
            lineList = line.split()
            final_info.append([lineList[0]]+map(int,lineList[1:3])+ [0])
    with open(output_bed,'r') as f:
        d = {subgenome: i+1 for i, subgenome in enumerate(set([line.split()[-1] for line in f.read().splitlines()]))}
        f.seek(0)
        for line in f.read().splitlines():
            if line:
                lineList = line.split()
                final_info.append([lineList[0]]+ map(int,lineList[1:3])+[d[lineList[-1]]])
    final_info = pd.DataFrame(np.array(final_info),columns=['chr', 'start', 'end', 'classifiedSubgenome'])
    final_info.to_csv(outname,index=False)


@polycracker.command(name='get_density')
@click.option('-gff', '--gff_file', default = '', help='Input gff file.',show_default=True)
@click.option('-w','--window_length', default = 75000, help= 'Window length for histogram of gene density.' ,show_default=True)
@click.option('-fai', '--fai_file', help='Full path to original, nonchunked fai file.', type=click.Path(exists=False))
@click.option('-fout', '--outputfname', default='gene_density.csv', help='Output csv file name.', show_default=True, type=click.Path(exists=False))
def get_density(gff_file,window_length,fai_file, outputfname):
    """Return gene or repeat density information in csv file from an input gff file. For shinyCircos."""
    subprocess.call('cut -f 1-2 %s > %s.genome'%(fai_file, 'genome'),shell=True)
    subprocess.call('bedtools makewindows -g genome.genome -w %d > windows.bed'%window_length,shell=True)
    subprocess.call('bedtools coverage -a windows.bed -b %s > coverage.bed'%(gff_file),shell=True)
    density_info = []
    with open('coverage.bed','r') as f:
        for line in f:
            if line and line.startswith('all') == 0:
                lineList = line.split()
                density_info.append([lineList[0]]+map(int,lineList[1:3])+[float(lineList[-1])])
    density_info = pd.DataFrame(density_info,columns=['chr','start','end','density'])
    density_info.to_csv(outputfname,index=False)


@polycracker.command(name='link2color')
@click.option('-csv', '--link_csv', help='Full path to link csv file.', type=click.Path(exists=False))
def link2color(link_csv):
    """Add color information to link file for shinyCircos."""
    outfname = link_csv.replace('.csv','.color.csv')
    df = pd.read_csv(link_csv)
    chrom2color = {chrom : chr(i+97) for i, chrom in enumerate(set(df['seg1'].as_matrix().tolist()))}
    df['color'] = np.vectorize(lambda x: chrom2color[x])(df['seg1'])
    df.to_csv(outfname,index=False)


@polycracker.command(name='multicol2multifiles')
@click.option('-csv', '--multi_column_csv', help='Full path to multi-column csv file, likely containing list of differential kmer counts.', type=click.Path(exists=False))
def multicol2multifiles(multi_column_csv):
    """Take matrix of total differential kmer counts, or similar matrix and break them up into single column files by found genome. For shinyCircos usage."""
    df = pd.read_csv(multi_column_csv)
    df_col = list(df.columns.values)
    for col in df_col[3:]:
        df[df_col[0:3]+[col]].to_csv(multi_column_csv.replace('.csv','.%s.csv'%col.strip(' ')),index=False)


@polycracker.command(name='count_repetitive')
@click.option('-fi', '--fasta_in', help='Fasta input file.', type=click.Path(exists=False))
def count_repetitive(fasta_in):
    """Infer percent of repetitive sequence in softmasked assembly"""
    import re
    f = Fasta(fasta_in)
    all_sequence = reduce(lambda x,y: x+y,map(lambda k: str(f[k][:]),f.keys())) # .replace('N','')
    lower_letters = len(re.findall(r'[a-z]',all_sequence))
    print(len(all_sequence),lower_letters,float(lower_letters)/len(all_sequence))


@polycracker.command(name='extract_sequences')
@click.option('-s', '--sequences_dict', default='S:S1,S2,S3-D:D1,D3,D10', show_default=True, help='Dictionary of sequences and which subgenome/genome to send them to.', type=click.Path(exists=False))
@click.option('-fi', '--fasta_in', help='Fasta input file.', type=click.Path(exists=False))
def extract_sequences(sequences_dict,fasta_in):
    """Extract sequences from fasta file and move them to new files as specified."""
    sequences_dict = {fasta_in[:fasta_in.rfind('.')]+'_%s'%genome_new+fasta_in[fasta_in.rfind('.'):]:sequences.split(',') for genome_new,sequences in [tuple(mapping.split(':')) for mapping in sequences_dict.split('-')]}
    for genome_new in sequences_dict:
        subprocess.call('samtools faidx %s %s > %s'%(fasta_in,' '.join(sequences_dict[genome_new]),genome_new),shell=True)


def find_genomic_space(scaffolds):
    dict_space = OrderedDict()
    for i,scaffold in enumerate(scaffolds.tolist()):
        sl = scaffold.split('_')
        dict_space[scaffold] = {}
        dict_space[scaffold]['_'.join(sl[:-2])] = np.mean(map(int,sl[-2:]))
    print dict_space
    scaffold_origin = {scaff:i for i, scaff in enumerate(set([dict_space[scaffold].keys()[0] for scaffold in dict_space.keys()]))}
    #print list(enumerate([dict_space[scaffold].keys()[0] for scaffold in dict_space.keys()]))
    dok = sps.dok_matrix((len(scaffolds),len(scaffold_origin.keys())),dtype=np.float)
    for i,scaffold in enumerate(dict_space.keys()):
        for scaff in dict_space[scaffold]:
            #print i,scaffold,scaff
            dok[i,scaffold_origin[scaff]] = dict_space[scaffold][scaff]
    dok = dok.tocsr()
    dok.data += 1000000000000.
    dok = dok.todense()
    #dok[dok==0] = np.max(dok) # 100000000000
    return dok


@polycracker.command(name="convert_subgenome_output_to_pickle")
@click.option('-id', '--input_dir', default='./', help='Directory containing solely the cluster/classified outputs, with names stored in txt files. Can input bed file here as well.', show_default=True, type=click.Path(exists=False))
@click.option('-s', '--scaffolds_pickle', default='scaffolds.p', help='Path to scaffolds pickle file.', show_default=True, type=click.Path(exists=False))
@click.option('-o', '--output_pickle', default='colors_pickle.p', help='Desired output pickle name. Can feed colors_pickle.p, or name you choose, into plotPositions.', show_default=True, type=click.Path(exists=False))
def convert_subgenome_output_to_pickle(input_dir, scaffolds_pickle, output_pickle):
    """Find cluster labels for all clusters/subgenomes in directory."""
    scaffolds = pickle.load(open(scaffolds_pickle,'rb'))
    clusters = defaultdict(list)
    if input_dir.endswith('.bed'):
        clusters = dict(zip(os.popen("awk '{print $1 \"_\" $2 \"_\" $3}' %s"%input_dir).read().splitlines(),os.popen("awk '{print $4}' %s"%input_dir).read().splitlines()))
        """
        with open(input_dir,'r') as f:
            for line in f:
                lineList = line.split()
                clusters['_'.join(lineList[0:3])] = lineList[-1].strip('\n')"""
    else:
        for file in os.listdir(input_dir):
            if file.endswith('.txt'):
                cluster_name = file.replace('.txt','')
                with open(input_dir+file,'r') as f:
                    for line in f:
                        clusters[line.strip('\n')] = cluster_name
    pickle.dump(np.vectorize(lambda x: clusters[x] if x in clusters.keys() else 'ambiguous')(scaffolds),open(output_pickle,'wb'))


@polycracker.command()
@click.option('-s', '--scaffolds_pickle', default='scaffolds.p', help='Path to scaffolds pickle file.', show_default=True, type=click.Path(exists=False))
@click.option('-o', '--output_pickle', default='colors_pickle.p', help='Path to output species pickle file.', show_default=True, type=click.Path(exists=False))
def species_comparison_scaffold2colors(scaffolds_pickle,output_pickle):
    """Generate color pickle file for plotPositions if list of scaffolds contains progenitors/species of origin. Useful for testing purposes."""
    scaffolds = pickle.load(open(scaffolds_pickle,'rb'))
    pickle.dump(np.vectorize(lambda x: x.split('_')[0])(scaffolds),open(output_pickle,'wb'))

@polycracker.command()
@click.pass_context
@click.option('-b3', '--bed3_directory', default='./', help='Directory containing bed3 files produced from subgenomeExtraction.', show_default=True, type=click.Path(exists=False))
@click.option('-o', '--original', default=0, help='Select 1 if subgenomeExtraction was done based on original nonchunked genome.', show_default=True)
@click.option('-sl', '--split_length', default=75000, help='Length of intervals in bedgraph files.', show_default=True)
@click.option('-fai', '--fai_file', help='Full path to original, nonchunked fai file.', type=click.Path(exists=False))
@click.option('-w', '--work_folder', default='./', help='Working directory for processing subgenome bedgraphs and unionbedgraph.', show_default=True, type=click.Path(exists=False))
def generate_unionbed(ctx, bed3_directory, original, split_length, fai_file, work_folder): # FIXME add smoothing function by overlapping the intervals
    """Generate a unionbedgraph with intervals of a specified length based on bed3 files from subgenomeExtraction"""
    global inputStr, key
    if work_folder.endswith('/') == 0:
        work_folder += '/'
    def grabLine(positions):
        global inputStr
        global key
        if positions[-1] != 'end':
            return '%s\t%s\t%s\n'%tuple([key]+map(str,positions[0:2]))
        else:
            return '%s\t%s\t%s\n'%(key, str(positions[0]), str(inputStr))
    def split(inputStr=0,width=60):
        if inputStr:
            positions = np.arange(0,inputStr,width)
            posFinal = []
            if inputStr > width:
                for i in range(len(positions[:-1])):
                    posFinal += [(positions[i],positions[i+1])]
            posFinal += [(positions[-1],'end')]
            splitLines = map(grabLine,posFinal)
            return splitLines
        else:
            return ''
    bedText = []
    for key, seqLength in [tuple(line.split('\t')[0:2]) for line in open(fai_file,'r') if line]:
        inputStr = int(seqLength)
        bedText += split(inputStr,split_length)
    a = BedTool('\n'.join(bedText),from_string=True).sort().saveas(work_folder + '/windows.bed') # add ability to overlap windows
    # convert coordinates of bed, use original fai_file
    bed_graphs_fn = []
    subprocess.call('cut -f 1-2 %s > %s.genome'%(fai_file, work_folder + '/genome'),shell=True)
    for bedfile in os.listdir(bed3_directory):
        if bedfile.endswith('.bed3') or bedfile.endswith('.bed'):
            if original == 0:
                bedText = []
                with open(bed3_directory+'/'+bedfile,'r') as f:
                    for line in f:
                        if line:
                            lineList = line.split()
                            lineList2 = lineList[0].split('_')
                            lineList[1:] = map(lambda x: str(int(lineList2[-2])+ int(x.strip('\n'))), lineList[1:])
                            lineList[0] = '_'.join(lineList2[:-2])
                            bedText.append('\t'.join(lineList))
                b = BedTool('\n'.join(bedText),from_string=True).sort()
            else:
                b = BedTool(bed3_directory+'/'+bedfile).sort()
            coverage = a.coverage(b).sort()
            bedgname = work_folder + bedfile[:bedfile.rfind('.')] + '.sorted.cov.bedgraph'
            with open(bedgname,'w') as bedgo:
                for line in str(coverage).splitlines():
                    lineInList = line.split()
                    lineOutputList = [lineInList[0], int(lineInList[1]), int(lineInList[2]), int(lineInList[4]) ]
                    bedgo.write('%s\t%d\t%d\t%d\n'%tuple(lineOutputList))
            bed_graphs_fn.append(bedgname)
    if len(bed_graphs_fn) == 1:
        subprocess.call('scp %s %s'%(bedgname,work_folder+'subgenomes.union.bedgraph'),shell=True)
        with open(work_folder+'subgenomes.union.bedgraph','r') as f:
            bed_txt = f.read().splitlines()
    else:
        x = BedTool()
        result = x.union_bedgraphs(i=bed_graphs_fn, g=work_folder + '/genome.genome')
        result.saveas(work_folder+'original_coordinates.subgenomes.union.bedgraph')
        bed_txt = str(result).splitlines()
    with open(work_folder+'subgenomes.union.bedgraph','w') as f:
        for line in bed_txt:
            lineList = line.split()
            lineList[-1] = lineList[-1].strip('\n')
            lineList[0] = '_'.join(lineList[0:3])
            lineList[2] = str(int(lineList[2]) - int(lineList[1]))
            lineList[1] = '0'
            f.write('\t'.join(lineList)+'\n')


@polycracker.command(name='correct_kappa')
@click.pass_context
@click.option('-dict','--comparisons_dict', default='subgenome_1:progenitorA,subgenome_2:progenitorB,subgenome_3:speciesA' , show_default=True, help='MANDATORY: Comma delimited mapping of inferred subgenome names output from polycracker to Progenitor/Species labels.', type=click.Path(exists=False))
@click.option('-p0','--scaffolds_pickle', default='scaffolds_stats.p', show_default=True, help='Pickle file containing the original scaffold names.',  type=click.Path(exists=False))
@click.option('-p1','--polycracker_pickle', default='scaffolds_stats.poly.labels.p', show_default=True, help='Pickle file generated from final_stats. These are polycracker results.',  type=click.Path(exists=False))
@click.option('-p2','--progenitors_pickle', default='scaffolds_stats.progenitors.labels.p', show_default=True, help='Pickle file generated from final_stats. These are progenitor mapping results.',  type=click.Path(exists=False))
@click.option('-r', '--from_repeat', is_flag=True, help='From repeat subgenome extraction.')
@click.option('-w','--work_dir', default='./', show_default=True, help='Work directory.',  type=click.Path(exists=False))
def correct_kappa(ctx,comparisons_dict,scaffolds_pickle,polycracker_pickle,progenitors_pickle,from_repeat, work_dir):
    """Find corrected cohen's kappa score. Used to test final_stats."""
    from sklearn.metrics import cohen_kappa_score
    scaffold_len = lambda scaffold: int(scaffold.split('_')[-1])-int(scaffold.split('_')[-2])
    load = lambda f: pickle.load(open(f,'rb'))
    scaffolds = load(scaffolds_pickle)
    weights = np.vectorize(scaffold_len)(scaffolds)
    weights_new = weights / float(np.sum(weights))
    polycracker2ProgSpec = {subgenome:progenitor_species for subgenome,progenitor_species in [tuple(mapping.split(':')) for mapping in comparisons_dict.split(',')]+[('ambiguous','ambiguous')]}
    if from_repeat:
        ctx.invoke(convert_subgenome_output_to_pickle,input_dir=polycracker_pickle, scaffolds_pickle=scaffolds_pickle, output_pickle=work_dir+'/scaffolds.repeats.labels.p')
        ctx.invoke(convert_subgenome_output_to_pickle,input_dir=progenitors_pickle, scaffolds_pickle=scaffolds_pickle, output_pickle=work_dir+'/progenitors.scaffolds.repeats.labels.p')
        polycracker_pickle = work_dir+'/scaffolds.repeats.labels.p'
        progenitors_pickle = work_dir+'/progenitors.scaffolds.repeats.labels.p'
    y_polycracker = np.vectorize(lambda x: polycracker2ProgSpec[x])(load(polycracker_pickle))
    y_progenitors = load(progenitors_pickle)
    a = y_progenitors != 'ambiguous'
    b = y_polycracker != 'ambiguous'
    sequence_in_common = (a & b)
    labels = np.unique(np.union1d(np.unique(y_progenitors),np.unique(y_polycracker)))
    cohen = cohen_kappa_score(y_progenitors[sequence_in_common],y_polycracker[sequence_in_common], sample_weight = weights_new[sequence_in_common])#,labels = labels[labels != 'ambiguous'])
    cohen2 = cohen_kappa_score(y_progenitors,y_polycracker,sample_weight = weights_new,labels = labels, weights=None)
    print classification_report(y_progenitors, y_polycracker, sample_weight=weights_new)
    print(weights[sequence_in_common].sum(),weights_new[sequence_in_common])
    print 'cohen unambig',cohen,'cohen all', cohen2


@polycracker.command(name='final_stats')
@click.option('-p/-s','--progenitors/--species',default=False, help='If running final_stats with progenitors without known subgenomes labelling, will compare progenitor results to polyCRACKER without ground truth. If separating species from sample or you already know subgenome labelling, --species will compare to ground truth.')
@click.option('-dict','--comparisons_dict', default='subgenome_1:progenitorA,subgenome_2:progenitorB,subgenome_3:speciesA' , show_default=True, help='MANDATORY: Comma delimited mapping of inferred subgenome names output from polycracker to Progenitor/Species labels.', type=click.Path(exists=False))
@click.option('-cbed','--correspondence_bed', default='correspondence.bed', show_default=True, help='Bed file containing the original scaffold names. Please check if this bed file contains all scaffolds, else generate file from splitFasta.',  type=click.Path(exists=False))
@click.option('-pbed','--polycracker_bed', default='subgenomes.bed', show_default=True, help='Bootstrap/cluster run results bed generated from outputgenerate_out_bed after running polycracker. These are polycracker results.',  type=click.Path(exists=False))
@click.option('-sbed','--progenitors_species_bed', default='progenitors.bed', show_default=True, help='Progenitors bed output from progenitorMapping or species bed output.',  type=click.Path(exists=False))
@click.option('-c','--cluster_analysis', is_flag=True, help='Whether to perform analyses on clustering results. Classification analysis will be disabled for this run if this option is selected, but you may need to add data from the above options.')
@click.option('-cs','--cluster_scaffold', default = 'scaffolds.p', help='Scaffolds pickle file used for cluster analysis.', show_default=True, type=click.Path(exists=False))
@click.option('-npy','--positions_npy',default='',help='Positions npy file if choosing to do clustering analysis. Will assess the validity of the clusters with a euclidean metric, so may not be as valid for non-convex sets. Leave blank if not running analysis.',type=click.Path(exists=False))
@click.option('-op','--original_progenitors',default='',help='Dictionary of original progenitor names to their fasta files. Eg. progenitorA:./progenitors/progenitorA.fasta,progenitorR:./progenitorsR.fasta')
@click.option('-n','--note',default='',help='Note to self that can be put into report.',type=click.Path(exists=False))
@click.pass_context
def final_stats(ctx,progenitors, comparisons_dict, correspondence_bed, polycracker_bed, progenitors_species_bed,cluster_analysis,cluster_scaffold,positions_npy,original_progenitors,note):
    """Analyzes the accuracy and agreement between polyCRACKER and ground truth species labels or progenitor mapped labels. Also compares polyCRACKER output to sizes of known progenitors. In future, may attempt to find similarity etween polyCRACKER sequences and progenitors through mash/bbsketch."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from sklearn.metrics import cohen_kappa_score,fowlkes_mallows_score,calinski_harabaz_score
    scaffold_len = lambda scaffold: int(scaffold.split('_')[-1])-int(scaffold.split('_')[-2])
    total_len = lambda scaffolds_list: sum([scaffold_len(scaffold) for scaffold in scaffolds_list if scaffold])
    if cluster_analysis == 0:
        # grab total length of all sequences
        total_length = int(os.popen("awk '{s+=($3-$2)} END {printf \"%%.0f\", s}' %s"%correspondence_bed).read())
        all_sequences = np.array(os.popen("awk '{print $1 \"_\" $2 \"_\" $3}' %s"%correspondence_bed).read().splitlines())
        pickle.dump(all_sequences,open('scaffolds_stats.p','wb'))
        polycracker2ProgSpec = {subgenome:progenitor_species for subgenome,progenitor_species in [tuple(mapping.split(':')) for mapping in comparisons_dict.split(',')]}
        weights = np.vectorize(scaffold_len)(all_sequences)
        weights_new = weights / float(np.sum(weights))
        ctx.invoke(convert_subgenome_output_to_pickle,input_dir=polycracker_bed, scaffolds_pickle='scaffolds_stats.p', output_pickle='scaffolds_stats.poly.labels.p')
        y_polycracker = pickle.load(open('scaffolds_stats.poly.labels.p','rb'))
        polycracker_unambiguous_total_len = sum(weights[y_polycracker != 'ambiguous'])
        polycracker_unambiguous_total_len_percent = float(polycracker_unambiguous_total_len)/total_length
        print(np.unique(polycracker2ProgSpec.keys()))
        print(polycracker2ProgSpec)
        print(y_polycracker)
        y_polycracker = np.vectorize(lambda x: polycracker2ProgSpec[x] if x in np.unique(polycracker2ProgSpec.keys()) else 'ambiguous')(y_polycracker)
        measures = {'Length: Total Genome':total_length,'Length: Total Poly': polycracker_unambiguous_total_len,'Ratio: [Total Poly]/[Total Genome]': polycracker_unambiguous_total_len_percent}
        if progenitors:
            ctx.invoke(convert_subgenome_output_to_pickle,input_dir=progenitors_species_bed, scaffolds_pickle='scaffolds_stats.p', output_pickle='scaffolds_stats.progenitors.labels.p')
            y_progenitors = pickle.load(open('scaffolds_stats.progenitors.labels.p','rb'))
            progenitor_unambiguous_total_len = sum(weights[y_progenitors != 'ambiguous'])
            progenitor_unambiguous_total_len_percent = float(progenitor_unambiguous_total_len)/total_length
            union = np.unique(np.union1d(np.unique(y_progenitors),np.unique(y_polycracker)))
            if 'ambiguous' in union:
                c_matrix_labels = np.hstack((np.setdiff1d(union,['ambiguous']),['ambiguous']))
            else:
                c_matrix_labels = union
            c_matrix = confusion_matrix(y_progenitors,y_polycracker,sample_weight=weights,labels=c_matrix_labels)
            sequence_agreement = (np.trace(c_matrix[:-1,:-1]) if 'ambiguous' in c_matrix_labels else np.trace(c_matrix))
            sequence_disagreement_n_ambiguous = total_length - sequence_agreement
            percent_agreement = float(sequence_agreement)/total_length
            percent_disagreement = float(sequence_disagreement_n_ambiguous)/total_length
            # other metrics, when both have nonambiguous sequences
            a = y_progenitors != 'ambiguous'
            b = y_polycracker != 'ambiguous'
            sequence_in_common = (a & b)
            cohen = cohen_kappa_score(y_progenitors[sequence_in_common],y_polycracker[sequence_in_common], sample_weight = weights_new[sequence_in_common],labels = np.setdiff1d(c_matrix_labels,['ambiguous']))
            cohen2 = cohen_kappa_score(y_progenitors,y_polycracker,sample_weight = weights_new,labels = c_matrix_labels)
            measures.update({'Length: Total Prog Map': progenitor_unambiguous_total_len,'Ratio: [Total Prog Map]/[Total Genome]': progenitor_unambiguous_total_len_percent,
                      'Length: Total Poly-Prog_Map Agreement':sequence_agreement, 'Ratio: [Poly-Prog_Map Agreement]/[Total Genome]':percent_agreement,
                      "Metric: Cohen's Kappa Unambiguous": cohen, "Metric: Cohen's Kappa All": cohen2, 'Metric: Jaccard Similarity Unambiguous':jaccard_similarity_score(y_progenitors[sequence_in_common],y_polycracker[sequence_in_common],normalize=True)})
            # FIXME ADD MORE AND PLOT C_MATRIX, send report to csv 'Ambiguous and Disagreement Sequence': sequence_disagreement_n_ambiguous,'Percent Disagreement over Total':percent_disagreement,
            if original_progenitors:
                progenitors_dict = {progenitor:original_progenitor for progenitor,original_progenitor in [tuple(mapping.split(':')) for mapping in original_progenitors.split(',')]}
                original_progenitor_lengths = {}
                for progenitor,original_progenitor in progenitors_dict.items():
                    subprocess.call('samtools faidx %s'%original_progenitor,shell=True)
                    original_progenitor_lengths[progenitor]=int(os.popen("awk '{s+=($2)} END {printf \"%%.0f\", s}' %s"%original_progenitor+'.fai').read())
                    measures['Length: Actual Prog %s'%progenitor] = original_progenitor_lengths[progenitor]
                for i in range((len(c_matrix_labels)-1 if 'ambiguous' in c_matrix_labels else len(c_matrix_labels))):
                    progenitor = c_matrix_labels[i]
                    measures['Length: %s Poly-Prog_Map Agreement'%progenitor]=c_matrix[i,i]
                    prog_sum = np.sum(c_matrix[i,:])
                    poly_sum = np.sum(c_matrix[:,i])
                    measures.update({'Length: %s Prog Map'%progenitor:prog_sum,'Length: %s Poly'%progenitor:poly_sum,
                                     'Ratio: [%s Prog Map]/[%s Actual Prog]'%(progenitor,progenitor):float(prog_sum)/original_progenitor_lengths[progenitor],
                                     'Ratio: [%s Poly]/[%s Actual Prog]'%(progenitor,progenitor):float(poly_sum)/original_progenitor_lengths[progenitor],
                                     'Ratio: [%s Poly]/[%s Prog Map]'%(progenitor,progenitor):float(poly_sum)/float(prog_sum)})
                measures.update({'Ratio: [Total Poly]/[Total Actual Prog]':float(polycracker_unambiguous_total_len)/sum(original_progenitor_lengths.values()),
                                 'Ratio: [Total Prog Map]/[Total Actual Prog]':float(progenitor_unambiguous_total_len)/sum(original_progenitor_lengths.values()),
                                 'Ratio: [Total Poly]/[Total Prog Map]':float(polycracker_unambiguous_total_len)/float(progenitor_unambiguous_total_len)})
            plot_labels = ('Classification Comparison','Progenitors','PolyCRACKER',c_matrix_labels)
        else:
            ctx.invoke(species_comparison_scaffold2colors,scaffolds_pickle='scaffolds_stats.p',output_pickle='scaffolds_stats.species.labels.p')
            y_species = pickle.load(open('scaffolds_stats.species.labels.p','rb')) # no ambiguous sequences here
            c_matrix_labels = np.unique(y_species)
            c_matrix = confusion_matrix(y_species,y_polycracker,sample_weight=weights,labels=c_matrix_labels)
            sequence_correctly_classified = np.trace(c_matrix)
            percent_correct = float(sequence_correctly_classified)/total_length
            # incorrect sequence is also ambiguous
            sequence_incorrectly_classified = total_length - sequence_correctly_classified
            percent_incorrect = float(sequence_incorrectly_classified)/total_length
            species_lengths = {species:sum(weights[y_species==species]) for species in np.unique(y_species)}
            polycracker_lengths = {species:sum(weights[y_polycracker==species]) for species in np.unique(y_species)}
            for species in np.unique(y_species):
                measures['Length: %s Original'%species] = species_lengths[species]
                measures['Length: %s Poly'%species] = polycracker_lengths[species]
                measures['Ratio: [%s Poly]/[%s Original]'%(species,species)] = float(polycracker_lengths[species])/species_lengths[species]
            for i in range(len(c_matrix_labels)):
                species = c_matrix_labels[i]
                measures['Length: %s Poly Correct'%species]=c_matrix[i,i]
                measures['Ratio: [%s Poly Correct]/[%s Original]'%(species,species)] = float(c_matrix[i,i])/species_lengths[c_matrix_labels[i]]
            if c_matrix.shape == (2,2):
                tn, fp, fn, tp = c_matrix.ravel()
                measures.update({'Metric: TN':tn,'Metric: FP':fp,'Metric: FN':fn,'Metric: TP':tp})
            class_report = classification_report(y_species, y_polycracker, sample_weight=weights_new, output_dict=True)
            if 0:
                class_report_rows = []
                class_report_data = []
                for line in class_report.splitlines()[2:-1]:
                    if line:
                        ll = line.split()
                        class_report_rows.append(ll[0])
                        class_report_data.append(ll[1:-1])
                line = class_report.splitlines()[-1]
                ll = line.split()
                class_report_rows.append(''.join(ll[0:3]))
                class_report_data.append(ll[3:-1])
                class_report_data = np.array(class_report_data).astype(float)
            class_report_df = pd.DataFrame(class_report)#,index=class_report_rows,columns=['Precision','Recall','F1-score'])
            class_report_df.to_csv('polycracker.classification.report.csv')
            plt.figure()
            sns.heatmap(class_report_df,annot=True)
            plt.savefig('polycracker.classification.report.png',dpi=300)
            measures.update({'Length: Total Poly Correct':sequence_correctly_classified,'Ratio: [Total Poly Correct]/[Total Genomes]':percent_correct,
                             'Metric: Jaccard Similarity':jaccard_similarity_score(y_species,y_polycracker,normalize=True),
                             'Metric: Classification Report Summary Avgs':class_report})
                             #'Metric: Classification Report Summary Avgs':['Precision: %f'%class_report_data[-1,0],'Recall: %f'%class_report_data[-1,1],'F1-Score: %f'%class_report_data[-1,2]]})
            #'Sequence incorrectly classified':sequence_incorrectly_classified, 'Percent of all, incorrect':percent_incorrect,
            # other metrics                              'brier score loss':'xxx',#[brier_score_loss(y_species, y_polycracker, sample_weight=weights,pos_label=species) for species in np.unique(y_species)],
            """'hamming loss':hamming_loss(y_species,y_polycracker,labels=np.unique(y_species),sample_weight=weights),
                             'zero one loss':zero_one_loss(y_species, y_polycracker, sample_weight=weights),
                             'Accuracy':accuracy_score(y_species, y_polycracker, sample_weight=weights)
                             'f1':f1_score(y_species,y_polycracker,average='weighted',sample_weight=weights),
                             'Precision and Recall, Fscore Support':precision_recall_fscore_support(y_species,y_polycracker,average='weighted',sample_weight=weights),"""
            # FIXME ADD MORE METRICS AND PLOT C_MATRIX
            plot_labels = ('Classification Confusion Matrix','Species','PolyCRACKER',c_matrix_labels)
    else:
        if cluster_scaffold.endswith('.bed'):
            all_sequences = np.array(os.popen("awk '{print $1 \"_\" $2 \"_\" $3}' %s"%cluster_scaffold).read().splitlines())
            scaffolds = all_sequences
            cluster_scaffold = 'new_scaffolds.p'
            pickle.dump(all_sequences,open(cluster_scaffold,'wb'))
        else:
            all_sequences = pickle.load(open(cluster_scaffold,'rb'))
            scaffolds = np.array(pickle.load(open(cluster_scaffold,'rb')))
        weights = np.vectorize(scaffold_len)(all_sequences)
        ctx.invoke(convert_subgenome_output_to_pickle,input_dir=polycracker_bed, scaffolds_pickle=cluster_scaffold, output_pickle='scaffolds_stats.poly.labels.p')
        y_pred = pickle.load(open('scaffolds_stats.poly.labels.p','rb'))
        measures = {}
        total_length = int(os.popen("awk '{s+=($3-$2)} END {printf \"%%.0f\", s}' %s"%correspondence_bed).read())
        measures.update({'Length: Total Genome':total_length,'Length: Total Poly':total_len(scaffolds)})
        for cluster in np.unique(y_pred):
            measures['Length: %s Poly'%cluster] = total_len(scaffolds[y_pred==cluster])
        if progenitors_species_bed and os.path.exists(progenitors_species_bed) and os.stat(progenitors_species_bed).st_size > 0:
            # PROGENITORS VS SPECIES
            if progenitors:
                ctx.invoke(convert_subgenome_output_to_pickle,input_dir=progenitors_species_bed, scaffolds_pickle=cluster_scaffold, output_pickle='scaffolds_stats.progenitors.labels.p')
                y_true = pickle.load(open('scaffolds_stats.progenitors.labels.p','rb'))
                plot_labels = ('Clustering Comparison','Progenitors','PolyCRACKER')
                measures.update({'Metric: AMI':adjusted_mutual_info_score(y_true,y_pred)})
                for progenitor in np.unique(y_true):
                    measures['Length: %s Prog Map from Poly Selected'%progenitor] = total_len(scaffolds[y_true==progenitor])
            else:
                ctx.invoke(species_comparison_scaffold2colors,scaffolds_pickle=cluster_scaffold,output_pickle='scaffolds_stats.species.labels.p')
                y_true = pickle.load(open('scaffolds_stats.species.labels.p','rb'))
                plot_labels = ('Clustering Comparison','Species','PolyCRACKER')
                measures.update({'Metric: V_measure':homogeneity_completeness_v_measure(y_true,y_pred)})
                for species in np.unique(y_true):
                    measures['Length: %s Original from Poly Selected'%species] = total_len(scaffolds[y_true==species])
            # metrics
            measures.update({'Metric: Fowlkes Mallows Score':fowlkes_mallows_score(y_true,y_pred),'Metric: ARI':adjusted_rand_score(y_true,y_pred)})
            # 'homogeneity':homogeneity_score(y_true,y_pred),Completeness':completeness_score(y_true,y_pred) 'ARI':adjusted_rand_score(y_true,y_pred),
            c_matrix = pd.crosstab(y_true,y_pred,values=weights,aggfunc=sum)
        if positions_npy:
            X = np.load(positions_npy)
            measures.update({'Metric: Silhouette Score':silhouette_score(X,y_pred,metric='euclidean'),'Metric: Calinski Harabaz':calinski_harabaz_score(X,y_pred)})
    plt.figure(figsize=(7,7))
    if cluster_analysis == 0:
        classes = plot_labels[3]
        c_matrix = pd.DataFrame(c_matrix,classes,['PolyCRACKER:' + cls for cls in classes])
        #tick_marks = np.arange(len(classes))
        #plt.xticks(tick_marks, classes)
        #plt.yticks(tick_marks, classes)
        plt.title(plot_labels[0])
        plt.xlabel(plot_labels[2])
        plt.ylabel(plot_labels[1])
    try:
        sns.heatmap(c_matrix)
    except:
        pass
    plt.xticks(rotation=45)
    if note:
        measures['Note'] = note
    for measure in measures:
        if type(measures[measure]) != type([]) or type(measures[measure]) != type(tuple([])):
            measures[measure] = [measures[measure]]
    measures = OrderedDict(sorted(measures.items()))
    print(measures)
    pd.DataFrame(measures).T.to_csv('polycracker.stats.analysis.csv',index=True)
    try:
        c_matrix.to_csv('polycracker.stats.confusion.matrix.csv',index=True)
        plt.savefig('polycracker.stats.confusion.matrix.pdf',dpi=300)
    except:
        pass


@polycracker.command()
@click.option('-csv','--dist_mat', default= 'output.cmp.csv', help='Input distance matrix from sourmash or can use feature matrix kmer_master_count_matrix.csv with option -f T.', type=click.Path(exists=False))
@click.option('-t','--transform_algorithm', default='mds', help='Dimensionality reduction technique to use.', type=click.Choice(['mds','tsne','spectral']))
@click.option('-o', '--output_file_prefix', default = 'output', show_default=True, help='Output file prefix, default outputs output.heatmap.png and output.[transform_algorithm].html.', type=click.Path(exists=False) )
@click.option('-j', '--n_jobs', default=1, help='Number of jobs for distance matrix transformation.', show_default=True)
@click.option('-f','--feature_space', default='n', help='Matrix is in feature space (n_samples x n_features or vice versa); y transforms via assuming matrix is n_samples x n_features and T assumes matrix is n_features x n_samples.', type=click.Choice(['n','y','T']))
@click.option('-k', '--kmeans_clusters', default = 0, help= 'If number of chosen clusters is greater than 0, runs kmeans clustering with chosen number of clusters, and choose most representative samples of clusters.', show_default=True)
def plot_distance_matrix(dist_mat,transform_algorithm,output_file_prefix,n_jobs,feature_space,kmeans_clusters):
    """Perform dimensionality reduction on sourmash distance matrix or feature space matrix and plot samples in lower dimensional space and cluster."""
    from MulticoreTSNE import MulticoreTSNE as TSNE
    from sklearn.manifold import MDS
    import seaborn as sns
    from sklearn.cluster import KMeans
    df = pd.read_csv(dist_mat,index_col=(None if feature_space == 'n' else 0))
    sample_names = np.array(list(df))
    if feature_space == 'T':
        df = df.transpose()
    if feature_space == 'y' or feature_space == 'T':
        sample_names = np.array(list(df.index))
        df = pairwise_distances(df,metric='manhattan') # euclidean fixme high dimensions
    else:
        df.iloc[:,:] = 1.-df.as_matrix()
    plt.figure()
    sns.heatmap(df,xticklabels=False, yticklabels=False)
    plt.savefig(output_file_prefix+'.heatmap.png',dpi=300)
    transform_dict = {'spectral':SpectralEmbedding(n_components=3,n_jobs=n_jobs,affinity='precomputed'),'tsne':TSNE(n_components=3,n_jobs=n_jobs,metric='precomputed'), 'mds':MDS(n_components=3,n_jobs=n_jobs,dissimilarity='precomputed')}
    if transform_algorithm == 'spectral':
        nn = NearestNeighbors(n_neighbors=5,metric='precomputed')
        nn.fit(df)
        df = nn.kneighbors_graph(df).todense()
    t_data = transform_dict[transform_algorithm].fit_transform(df)
    py.plot(go.Figure(data=[go.Scatter3d(x=t_data[:,0],y=t_data[:,1],z=t_data[:,2],mode='markers',marker=dict(size=3),text=sample_names)]),filename='%s.%s.html'%(output_file_prefix,transform_algorithm),auto_open=False)
    if kmeans_clusters > 0:
        km = KMeans(n_clusters=kmeans_clusters)
        km.fit(t_data)
        centers = km.cluster_centers_
        df = pd.DataFrame(t_data,index=sample_names,columns=['x','y','z'])
        df['labels'] = km.labels_
        df['distance_from_center'] = np.linalg.norm(df.loc[:,['x','y','z']].as_matrix() - np.array([centers[label] for label in df['labels'].as_matrix()]),axis=1)
        plots = []
        representative_points = []
        for label in df['labels'].unique():
            dff = df[df['labels'] == label]
            representative_points.append(dff['distance_from_center'].argmin())
            plots.append(go.Scatter3d(x=dff['x'],y=dff['y'],z=dff['z'],name='Cluster %d'%label,text=list(dff.index),mode='markers',marker=dict(size=3)))
        plots.append(go.Scatter3d(x=centers[:,0],y=centers[:,1],z=centers[:,2],name='Cluster Centers',text=['Cluster %d'%label for label in df['labels'].unique()],mode='markers',marker=dict(size=6,opacity=0.5,color='purple')))
        with open('representative_points.txt','w') as f:
            f.write(','.join(representative_points)+'\nCopy Commands:\nscp '+ ' '.join(representative_points+['folder_of_choice']))
        py.plot(plots,auto_open=False,filename='%s.%s.clusters.html'%(output_file_prefix,transform_algorithm))


@click.pass_context
def jaccard_clustering(ctx, work_dir, sparse_diff_kmer_matrix, scaffolds_pickle, n_subgenomes, n_neighbors_scaffolds, weights):
    scaffolds = np.array(pickle.load(open(scaffolds_pickle,'rb')))
    data = sps.load_npz(sparse_diff_kmer_matrix)
    data.data[:] = 1
    data = data.toarray()
    nn_jaccard = NearestNeighbors(n_neighbors=n_neighbors_scaffolds,metric='jaccard')
    nn_jaccard.fit(data)
    jaccard_distance_NN_matrix = nn_jaccard.kneighbors_graph(data, mode = ('distance' if weights else 'connectivity'))
    sps.save_npz(work_dir+'jaccard_nn_graph.npz',jaccard_distance_NN_matrix)
    # FIXME Maybe use spectral clustering
    #agg_cluster = AgglomerativeClustering(n_clusters=n_subgenomes, linkage='average',affinity='precomputed', connectivity=jaccard_distance_NN_matrix)
    #agg_cluster.fit(jaccard_distance_NN_matrix.toarray())
    spec_cluster = SpectralClustering(n_clusters=n_subgenomes, affinity='precomputed')
    spec_cluster.fit(jaccard_distance_NN_matrix)
    try:
        os.makedirs(work_dir+'jaccard_clustering_results')
    except:
        pass
    labels = spec_cluster.labels_
    print len(scaffolds), len(labels)
    for label in set(labels):
        with open(work_dir+'jaccard_clustering_results/'+str(label)+'.txt','w') as f:
            f.write('\n'.join(list(scaffolds[labels==label])))
    cluster_labels = np.vectorize(lambda x: 'Subgenome %d'%x)(labels)
    pickle.dump(cluster_labels,open(work_dir+'jaccard_clustering_labels.p','wb'))
    ctx.invoke(plotPositions,positions_npy=work_dir+'transformed_diff_kmer_matrix.npy', labels_pickle=work_dir+'scaffolds_diff_kmer_analysis.p', colors_pickle=work_dir+'jaccard_clustering_labels.p', output_fname=work_dir+'jaccard_clustering_results/jaccard_plot.html', graph_file=work_dir+'jaccard_nn_graph.npz', layout='standard', iterations=25)


@polycracker.command()
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
def dash_genetic_algorithm_hdbscan_test(work_dir,kmer_count_matrix,reduction_method,n_dimensions,kernel,metric,min_cluster_size,min_samples,n_neighbors,hyperparameter_scan,min_number_clusters,low_counts,n_jobs,silhouette,validity):
    """Save labels of all hdbscan runs, generate distance matrix for input into dash app. Output distance matrix, dataframe of parameters and score, and matrix of labels that produce distance matrix, referenced by parameters"""
    import hdbscan
    from sklearn.manifold import SpectralEmbedding#, TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.metrics import calinski_harabaz_score, silhouette_score
    from evolutionary_search import maximize
    import seaborn as sns
    from sklearn.metrics import adjusted_mutual_info_score as AMI
    # FIXME mahalanobis arguments to pass V, debug distance metrics
    # FIXME add ECTD as a distance metric? Feed as precomputed into hdbscan?
    hdbscan_metric = (metric if metric not in ['ectd','cosine','mahalanobis'] else 'precomputed')
    all_labels = defaultdict(list)
    global count
    count = 0

    def cluster_data(t_data,min_cluster_size, min_samples, cluster_selection_method= 'eom'): # , n_neighbors = n_neighbors
        #print min_cluster_size, min_samples, kernel, n_neighbors
        labels = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples= min_samples, cluster_selection_method=cluster_selection_method, metric = hdbscan_metric, alpha = 1.0).fit_predict(t_data)
        #lp = LabelPropagation(kernel=kernel, n_neighbors = n_neighbors) # FIXME Try decision trees next, maybe just use highest chi square valued ones for training
        #lp.fit(t_data,labels) #kmer_count_matrix
        #labels = np.array(lp.predict(t_data))
        return labels

    scoring_method = lambda X, y: hdbscan.validity.validity_index(X,y,metric=hdbscan_metric) if validity else (silhouette_score(X,y,metric='precomputed' if metric =='mahalanobis' else 'mahalanobis',V=(np.cov(X,rowvar=False) if metric != 'mahalanobis' else '')) if silhouette else calinski_harabaz_score(X,y))

    def return_cluster_score(t_data,min_cluster_size, min_samples, cluster_selection_method): # , n_neighbors
        global count
        click.echo(' '.join(map(str,[min_cluster_size, min_samples, cluster_selection_method]))) # , n_neighbors
        labels = cluster_data(t_data,min_cluster_size, min_samples, cluster_selection_method) # , n_neighbors
        n_clusters = labels.max() + 1
        X = t_data if validity else t_data[labels != -1,:]
        y = labels if validity else labels[labels != -1]
        score = scoring_method(X,y)/((1. if n_clusters >= min_number_clusters else float(min_number_clusters - n_clusters + 1))*(1.+len(labels[labels == -1])/float(len(labels)))) #FIXME maybe change t_data[labels != -1,:], labels[labels != -1] and (1.+len(labels[labels == -1])/float(len(labels)))
        all_labels['%d_%d_%d_%s'%(count,min_cluster_size, min_samples, cluster_selection_method)]=[labels,score]
        count += 1
        return score

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
    #np.save(work_dir+'kmer_counts_pca_results.npy',KernelPCA(n_components=3).fit_transform(t_data) if t_data.shape[1] > 3 else t_data)
    pd.DataFrame(np.hstack((np.array(labels_text)[:,np.newaxis],KernelPCA(n_components=3).fit_transform(t_data) if t_data.shape[1] > 3 else t_data)),columns = ['labels','x','y','z']).to_csv(work_dir+'pca_output.csv')
    if metric == 'ectd':
        t_data = ectd_graph(t_data)
    elif metric == 'cosine':
        t_data = pairwise_distances(t_data,metric='cosine')
    elif metric == 'mahalanobis':
        t_data = pairwise_distances(t_data,metric='mahalanobis',V=np.cov(t_data,rowvar=True))
    if 1: # FIXME Test hyperparameter_scan, add ability to use label propagation, or scan for it or iterations of label propagation
        best_params, best_score, score_results, hist, log = maximize(return_cluster_score, dict(min_cluster_size = np.unique(np.linspace(low_counts[0],min_cluster_size,25).astype(int)).tolist(), min_samples = np.unique(np.linspace(low_counts[1],min_samples, 25).astype(int)).tolist(), cluster_selection_method= ['eom', 'leaf'] ), dict(t_data=t_data), verbose=True, n_jobs = 1, population_size=15, generations_number=7, gene_mutation_prob=0.3, gene_crossover_prob = 0.3)#15 7 fixme <--add these back, also begin to implement dash app, note: back up all final poly runs to disk... # n_neighbors = np.unique(np.linspace(low_counts[2], n_neighbors, 10).astype(int)).tolist()),
        labels = cluster_data(t_data,min_cluster_size=best_params['min_cluster_size'], min_samples=best_params['min_samples'], cluster_selection_method= best_params['cluster_selection_method']) # , n_neighbors = best_params['n_neighbors']
        print best_params, best_score, score_results, hist, log
    else:
        labels = cluster_data(t_data,min_cluster_size, min_samples)
    rules_orig = np.vectorize(lambda x: 'rule %d'%x)(labels)
    pickle.dump(rules_orig,open(work_dir+'rules.p','wb'))
    gen = defaultdict(list)
    indivs = np.array(all_labels.keys())
    indivs_vals = np.vectorize(lambda x: int(x[:x.find('_')]))(indivs)
    indivs = indivs[np.argsort(indivs_vals)]
    indivs_vals = indivs_vals[np.argsort(indivs_vals)]
    count = 0
    print list(enumerate(log))
    for i,generation in enumerate(log):
        print count
        gen[i] = indivs[count:count+generation['nevals']]#np.intersect1d(indivs_vals,np.arange(count,count+generation['nevals']))]
        count += generation['nevals']
    # FIXME generate dataframe
    # FIXME df = pd.DataFrame([[generation,int(parameters[:parameters.find('_')]),parameters,all_labels[parameters][1]]+parameters.split('_')[1:] for parameters in gen[generation] for generation in gen],columns = ['generation','individual','parameters','score','min_cluster_size', 'min_samples', 'cluster_selection_method'])
    # FIXME TypeError: unhashable type: 'dict'
    #print gen
    #print [gen[generation] for generation in gen]
    #print [all_labels[gen[generation][0]] for generation in gen]
    #print gen[0]
    #print all_labels
    #print [all_labels[parameters][1] for parameters in gen[0].tolist()]
    #print [all_labels[parameters][1] for parameters in gen[0]]
    generation_data = []
    for generation in gen:
        for parameters in gen[generation]:
            generation_data.append([generation,int(parameters[:parameters.find('_')]),parameters,all_labels[parameters][1]]+parameters.split('_')[1:])
    # [[generation,int(parameters[:parameters.find('_')]),parameters,all_labels[parameters][1]]+parameters.split('_')[1:] for parameters in gen[generation] for generation in gen]
    df = pd.DataFrame(generation_data,columns = ['generation','individual','parameters','score','min_cluster_size', 'min_samples', 'cluster_selection_method'])
    print df
    #pickle.dump(all_labels,open(work_dir+'all_labels.p','wb'))
    all_labels = pd.DataFrame(np.array([all_labels[indiv][0] for indiv in indivs]).T,columns = indivs)
    df.to_csv(work_dir+'generations.csv')
    all_labels.to_csv(work_dir+'db_labels.csv')
    distance_mat = pd.DataFrame(np.zeros((len(indivs),len(indivs))),index=indivs,columns=indivs)
    for i,j in list(combinations(indivs,2)):
        distance_mat.loc[i,j] = 1.-AMI(all_labels[i],all_labels[j])
        distance_mat.loc[j,i] = distance_mat.loc[i,j]
    distance_mat.to_csv(work_dir+'distance_mat.csv')
    #distance_mat = similarity_mat
    #distance_mat.iloc[:,:] = 1. - distance_mat.as_matrix()
    # FIXME feed distance matrix, generations, db_labels matrix, and pca matrix into dash app, subset by generation, score, etc... also kmer coordinate labels


@polycracker.command(name='bed2scaffolds_pickle')
@click.option('-i', '--bed_file', default = 'correspondence.bed', show_default=True, help='Input bed file, original coordinate system.', type=click.Path(exists=False))
@click.option('-o', '--output_pickle', default='scaffolds_new.p', help='Pickle file containing the new scaffolds.', show_default=True, type=click.Path(exists=False))
def bed2scaffolds_pickle(bed_file, output_pickle):
    """Convert correspondence bed file, containing complete scaffold information into a list of scaffolds file."""
    pickle.dump(np.array([line for line in os.popen("awk '{print $1 \"_\" $2 \"_\" $3 }' %s"%bed_file).read().splitlines() if line]),open(output_pickle,'wb'))


@polycracker.command(name='scaffolds2colors_specified')
@click.option('-s', '--scaffolds_pickle', default='scaffolds.p', help='Path to scaffolds pickle file.', show_default=True, type=click.Path(exists=False))
@click.option('-d', '--labels_dict', help='Comma delimited dictionary of text to find in scaffold to new label. Example: text1:label1,text2:label2')
@click.option('-o', '--output_pickle', default='colors_pickle.p', help='Pickle file containing the new scaffold labels.', show_default=True, type=click.Path(exists=False))
def scaffolds2colors_specified(scaffolds_pickle,labels_dict,output_pickle):
    """Attach labels to each scaffold for use of plotPositions. Colors scaffolds based on text within each scaffold. Useful for testing/evaluation purposes."""
    scaffolds = pickle.load(open(scaffolds_pickle,'rb'))
    labels_dict = {text:label for text,label in [tuple(mapping.split(':')) for mapping in labels_dict.split(',')]}
    def relabel(scaffold,labels_dict = labels_dict):
        for text in labels_dict:
            if text in scaffold:
                return labels_dict[text]
        return 'unlabelled'
    pickle.dump(np.vectorize(relabel)(scaffolds),open(output_pickle,'wb'))


@polycracker.command() # FIXME mash, sourmash, kWIP, bbsketch
def mash_test(split_fasta):
    """Sourmash integration in development."""
    print 'Under development'


##################################################################################

###################################### TEST ######################################

@polycracker.command()
def run_tests():
    """Run basic polyCRACKER tests to see if working properly."""
    subprocess.call('pytest -q test_polycracker.py',shell=True)

##################################################################################

###################################### MAIN ######################################

if __name__ == '__main__':
    polycracker()

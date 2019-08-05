import rpy2.robjects as robjects
#import rpy2.interactive as r
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects.lib.ggplot2 as ggplot2
import os
import click
import glob
import numpy as np, pandas as pd
pandas2ri.activate()
# http://web.mit.edu/~r/current/arch/i386_linux26/lib/R/library/GenomicRanges/html/makeGRangesFromDataFrame.html

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def plotter():
    pass


class PackageInstaller:
    def __init__(self):
        pass

    def install_bioconductor(self):
        base = importr('base')
        base.source("http://www.bioconductor.org/biocLite.R")

    def install_ggbio(self):
        biocinstaller = importr("BiocInstaller") # fixme
        biocinstaller.biocLite(robjects.vectors.StrVector(["GenomicRanges","ggbio","biovizBase"]))


class Karyogram:
    def __init__(self, chrom_sizes):
        self.chrom_sizes = dict(np.loadtxt(chrom_sizes,dtype=str).tolist())
        self.seqinfo_str = "c({})".format(','.join('{}={}'.format(k,v) for k,v in self.chrom_sizes.items()))
        self.density = False

    def add_cytoband(self, cytoband_file, custom=True):
        df = pd.read_csv(cytoband_file,header=None)
        df['chr'] = df[0]
        df['start'] = df[1].map(lambda x: x.split('-')[0]).astype(int)
        df['end'] = df[1].map(lambda x: x.split('-')[1]).astype(int)
        df['strand'] = '+'
        df['score'] = 0
        df = df[['chr','start','end','score']]
        GenomicRanges = importr("GenomicRanges")
        self.cytoband = GenomicRanges.makeGRangesFromDataFrame(pandas2ri.py2ri(df))

    def add_density(self, density_file, score_name = 'geneDensity', custom=True):
        if custom:
            df = pd.read_csv(density_file)
            df['chr'] = df['Chromosome']
            df['start'] = df['Bin'].map(lambda x: x.split('-')[0]).astype(int)
            df['end'] = df['Bin'].map(lambda x: x.split('-')[1]).astype(int)
            df['score'] = df[score_name].astype(float)
            df['strand'] = '+'
            df = df[['chr','start','end','score']]
            GenomicRanges = importr("GenomicRanges")
            self.density = GenomicRanges.makeGRangesFromDataFrame(pandas2ri.py2ri(df),keep_extra_columns=True)#r('function(df) {with(df,GRanges(chr, IRanges(start, end), strand, score))}')(pandas2ri.py2ri(df))#
        else:
            self.density=False

    def plot(self, outputfilename, cytoband=False):
        #             if cytoband:
        #      self.cytoband
        ggbio = importr("ggbio")
        #r('getOption("biovizBase")$cytobandColor')
        seq_lengths = r(self.seqinfo_str)
        if self.density:
            self.density.slots['seqlengths'] = seq_lengths
            r('seqlengths')(self.density)
            print(self.density)
            plt = r("function(g) {autoplot(g, layout='karyogram', cytoband=F, aes(color=score))}")(self.density) #+ ggplot2.aes_string(colour='score')
            if cytoband:
                plt+=ggbio.autoplot(self.cytoband, layout='karyogram', cytoband=True)
            ggbio.ggsave(outputfilename)#,plot=plt)

#### COMMANDS ####

## Install ##
@plotter.command()
def install_bioconductor():
    installer = PackageInstaller()
    installer.install_bioconductor()

@plotter.command()
def install_ggbio():
    installer = PackageInstaller()
    installer.install_ggbio()

## Plot ##

@plotter.command()
@click.option('-c', '--chrom_sizes', default='./ABR113.chrom.sizes', help='Chromosome sizes tsv.', type=click.Path(exists=False), show_default=True)
@click.option('-d', '--density_file', default='./geneDensity.txt', help='Gene Density csv.', type=click.Path(exists=False), show_default=True)
@click.option('-s', '--score_name', default='geneDensity', help='Field for score acquisition.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--outputfilename', default='./karyogram.png', help='Output image.', type=click.Path(exists=False), show_default=True)
@click.option('-cy', '--cytoband', default='', help='Cytoband info.', type=click.Path(exists=False), show_default=True)
def plot_density_track(chrom_sizes, density_file, score_name, outputfilename, cytoband):
    karyogram = Karyogram(chrom_sizes)
    karyogram.add_density(density_file, score_name)
    if cytoband:
        karyogram.add_cytoband(cytoband)
    karyogram.plot(outputfilename, cytoband)

#################

if __name__ == '__main__':
    plotter()

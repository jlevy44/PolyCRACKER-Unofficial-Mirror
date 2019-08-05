#!/usr/bin/env nextflow

//////////////////////////////////////////////////////////////////////////////////
// parsing functions

String findValue(String inputStr) {
configFile = file('config_polyCRACKER.txt')
allLines = configFile.readLines()
for( line in allLines ) {
    if(line.startsWith(inputStr)){
        println (line - inputStr - '= ')
        return (line - inputStr - '= ' - ' ');
    }
}

}

String findPeak(String peakfname){

    //String[] parts = peakfname.split("_");
    //String peakfinalname = parts[0];
    int finalPosition = peakfname.lastIndexOf('_');
    String peakfinalname = peakfname.substring(0,finalPosition);
    return peakfinalname;

}

//////////////////////////////////////////////////////////////////////////////////
// parse config_file

pythonPath = findValue('pythonPath ' );
systemPath = findValue('systemPath ' );
blastPath = findValue('blastPath ' );
kmercountPath = findValue('kmercountPath ' );
fastaPath = findValue('fastaPath ' );
bedPath = findValue('bedPath ' );
sortPath = findValue('sortPath ' );

blastMemory = findValue('blastMemory ' );
threads = findValue('threads ' );
genome = findValue('genome ' );
BBstr = findValue('BB ');
n_subgenomes = findValue('n_subgenomes ');
n_dimensions = findValue('n_dimensions ');
splitLength = findValue('splitFastaLineLength ');
bootstrap = findValue('bootstrap ');
kmerLength = findValue('kmerLength ');
k_search_length = findValue('k_search_length ');
transformMetric = findValue('transformMetric ');
n_neighbors = findValue('n_neighbors ');
weighted_nn = findValue('weighted_nn ');
metric = findValue('metric ');
kmer_low_count = findValue('kmer_low_count ');
diff_kmer_threshold = findValue('diff_kmer_threshold ');
unionbed_threshold = findValue('unionbed_threshold ');
minChunkSize = findValue('minChunkSize ');
removeNonChunk = findValue('removeNonChunk ');
minChunkThreshold = findValue('minChunkThreshold ');
lowMemory = findValue('lowMemory ');
highBool = findValue('use_high_count ');
tfidf = findValue('tfidf ');
perfect_mode = findValue('perfect_mode ');
mst = findValue('mst ');
kmer_high_count = findValue('kmer_high_count ');
sampling_sensitivity = findValue('sampling_sensitivity ');
preFilter = findValue('preFilter ');
grabAll = findValue('grabAllClusters ');
diff_sample_rate = findValue('diff_sample_rate ');
default_kmercount_value = findValue('default_kmercount_value ');

reduction_techniques = findValue('reduction_techniques ').split(',');
clusterModels = findValue('clusterMethods ').split(',');

BB = findValue('BB ').asType(Integer);
splitFast = findValue('splitFasta ').asType(Integer);
writeKmer = findValue('writeKmer ').asType(Integer);
fromFasta = findValue('kmer2Fasta ').asType(Integer);
original = findValue('original ').asType(Integer);
writeBlast = findValue('writeBlast ').asType(Integer);
runBlastParallel = findValue('runBlastParallel ').asType(Integer);
b2b = findValue('blast2bed ').asType(Integer);
genMat = findValue('generateClusteringMatrix ').asType(Integer);
trans = findValue('transformData ').asType(Integer);
clust = findValue('ClusterAll ').asType(Integer);
extract = findValue('extract ').asType(Integer);

genomeSplitName = genome - '_split' - '.fasta' - '.fa' + '_split.fa';
blastDBName = genomeSplitName - '.fa';
genomeFullPath = fastaPath + genomeSplitName;
n_clusters = (n_subgenomes.asType(Integer) + 1).asType(String);
originalStr = original.asType(String);
blastMemStr = "export _JAVA_OPTIONS='-Xms5G -Xmx" + blastMemory + "G'"

kmercountName = genomeSplitName - '.fa' + '.kcount' + '.fa';
blastName = kmercountName - '.fa' + '.BLASTtsv.txt';
workingDir = new File('').getAbsolutePath();

if ( original == 1){
    genome2 = genome - '\n';
    blastDBName2 = genome - '\n' - '.fasta' -'.fa';
    originalGenome = fastaPath + genome;
}
else {
    genome2 = genomeSplitName;
    blastDBName2 = blastDBName;
    originalGenome = genomeFullPath;
}

//////////////////////////////////////////////////////////////////////////////////
// split polyploid fasta file

genomeChan = Channel.from(genome - '\n')
genomeChan2 = Channel.create()

process splitFastaProcess {
    executor = 'local'
    cpus = { splitFast == 1 ? 2 : 1 }

input:
    val genomeName from genomeChan

output:
    stdout genomeChan2

script:
    if(splitFast == 1)
        """
        #!/bin/bash
        cd ${workingDir}
        polycracker splitFasta --fasta_file=${genomeName} --fasta_path=${fastaPath} --chunk_size=${splitLength} --prefilter=${preFilter} --min_chunk_size=${minChunkSize} --remove_non_chunk=${removeNonChunk} --min_chunk_threshold=${minChunkThreshold} --low_memory=${lowMemory}
        """
    else
        """
        echo ${genomeSplitName}
        """

}

//////////////////////////////////////////////////////////////////////////////////
// write kmer counts

genomeChan3 = genomeChan2.last()
genomeChan4 = Channel.create()

process writeKmerCount {
    cpus = { writeKmer == 1 ? 4 : 1 }

input:
    val genomeName from genomeChan3

output:
    val genomeName into genomeChan4

script:
    if(writeKmer == 1)
        """
        #!/bin/bash
        echo ${workingDir}
        cd ${workingDir}
        echo polycracker writeKmerCount --fasta_path=${fastaPath} --kmercount_path=${kmercountPath} --kmer_length=${kmerLength} --blast_mem=${blastMemory}
        polycracker writeKmerCount --fasta_path=${fastaPath} --kmercount_path=${kmercountPath} --kmer_length=${kmerLength} --blast_mem=${blastMemory}
        """
    else
        """
        touch done
        """

}

//////////////////////////////////////////////////////////////////////////////////
// convert kmercount files into fasta files for blasting

genomeChan5 = Channel.create()
genomeChan55 = Channel.create()

process kmer2Fasta {
    executor = 'local'
    cpus = { fromFasta == 1 ? 2 : 1 }

input:
    val genomeName from genomeChan4

output:
    val genomeName into genomeChan5
    val genomeName into genomeChan55

script:
    if(fromFasta == 1 && BB == 0)
        """
        #!/bin/bash
        cd ${workingDir}
        module load blast+/2.6.0
        polycracker kmer2Fasta --kmercount_path=${kmercountPath} --kmer_low_count=${kmer_low_count} --high_bool=${highBool} --kmer_high_count=${kmer_high_count} --sampling_sensitivity=${sampling_sensitivity}
        ${blastMemStr} && makeblastdb -in ${genomeFullPath} -dbtype nucl -out ${blastDBName}.blast_db
        """
    else if(fromFasta == 1 && BB == 1)
        """
        #!/bin/bash
        cd ${workingDir}
        polycracker kmer2Fasta --kmercount_path=${kmercountPath} --kmer_low_count=${kmer_low_count} --high_bool=${highBool} --kmer_high_count=${kmer_high_count} --sampling_sensitivity=${sampling_sensitivity}
        ${blastMemStr} && bbmap.sh ref=${genomeFullPath} k=${k_search_length}
        """
    else
        """
        touch done
        """

}


//////////////////////////////////////////////////////////////////////////////////
// create a reference blast/bbmap database for future blast queries


process createOrigDB {
    executor = 'local'
    cpus = { original == 1 ? 2 : 1 }

input:
    val genomeName from genomeChan55

script:
    if(original == 1 && BB == 0)
        """
        #!/bin/bash
        cd ${workingDir}
        module load blast+/2.6.0
        ${blastMemStr} && makeblastdb -in ${originalGenome} -dbtype nucl -out ${blastDBName2}.blast_db
        """
    else if(original == 1 && BB == 1)
        """
        #!/bin/bash
        cd ${workingDir}
        ${blastMemStr} && bbmap.sh ref=${originalGenome} k=${k_search_length}
        """
    else
        """
        touch done
        """

}

//////////////////////////////////////////////////////////////////////////////////
// create a reference blast/bbmap database for future blast queries

// if running multiple blast jobs in parallel for large file sizes/memory/time issues
if(runBlastParallel == 1){
    genomeChan5.map{it -> file(kmercountPath+kmercountName)}
               .splitFasta(by: 50000,file: true)
               .set {kmerFasta}
    }
else{
    genomeChan5.map{it -> file(workingDir + '/' + kmercountPath+kmercountName)}
               .set {kmerFasta}
}

blast_result = Channel.create()

process BlastOff {
    cpus = { writeBlast == 1 ? 4 : 1 }

input:
    file 'query.fa' from kmerFasta //'query.fa'

output:
    file blast_result

script:
    if(writeBlast == 1)
        """
        #!/bin/bash
        cwd=\$(pwd)
        cd ${workingDir}
        polycracker blast_kmers -m ${blastMemory} -t ${threads} -r ${workingDir}/${genomeFullPath} -k \$cwd/query.fa -o results.sam -pm ${perfect_mode} -kl ${k_search_length}
        cd -
        mv ${workingDir}/results.sam blast_result
        """
    else
        """
        touch blast_result
        """

}

// concatenating blast results together
if (writeBlast == 1){
blast_result.collectFile(name: blastPath + blastName)
            .map {file -> genomeSplitName}
            .set {genomeChan6}
            }
else {
blast_result.collectFile()
            .map {file -> genomeSplitName}
            .set {genomeChan6}
}

//////////////////////////////////////////////////////////////////////////////////
// convert blast results into a bed file

genomeChan7 = Channel.create()

process blast2bed {
    executor = 'local'
    cpus = { b2b == 1 ? 2 : 1 }

input:
    val genomeName from genomeChan6

output:
    val genomeName into genomeChan7


script:
    if(b2b == 1)
        """
        #!/bin/bash
        cd ${workingDir}
        polycracker blast2bed --blast_file=${blastPath}${blastName} --bb=1 --low_memory=${lowMemory}
        """
    else
        """
        touch done
        """

}

//////////////////////////////////////////////////////////////////////////////////
// generates a sparse kmer-matrix from the kmer blasted bed file

genomeChan8 = Channel.create()

process genClusterMatrix_kmerPrevalence {
    cpus = { genMat == 1 ? 2 : 1 }

input:
    val genomeName from genomeChan7

output:
    val genomeName into genomeChan8


script:
    if(genMat == 1)
        """
        #!/bin/bash
        cd ${workingDir}
        polycracker generate_Kmer_Matrix --kmercount_path=${kmercountPath} --genome=${genomeName} --chunk_size=${splitLength} --min_chunk_size=${minChunkSize} --remove_non_chunk=${removeNonChunk} --min_chunk_threshold=${minChunkThreshold} --low_memory=${lowMemory} --prefilter=${preFilter} --fasta_path=${fastaPath} --genome_split_name=${genomeSplitName} --original_genome=${genome} --tfidf=${tfidf}
        """
    else
        """
        touch done
        """

}

//////////////////////////////////////////////////////////////////////////////////
// Perform dimensionality reduction on sparse kmer matrix

process transform {
    executor = 'local'
    cpus = { trans == 1 ? 2 : 1 }

input:
    val genomeName from genomeChan8
    each technique from reduction_techniques

output:
    file 'test.txt' into transformedData

script:
    if(trans == 1)
        """
        #!/bin/bash
        cd ${workingDir}
        polycracker transform_plot --technique=${technique} --n_subgenomes=${n_subgenomes} --metric=${transformMetric} --n_dimensions=${n_dimensions} --tfidf=${tfidf}
        cd -
        echo main_${technique}_${n_subgenomes}_transformed3D.npy > test.txt
        """
    else
        """
        #!/bin/bash
        cd ${workingDir}
        touch main_${technique}_${n_subgenomes}_transformed3D.npy
        cd -
        echo main_${technique}_${n_subgenomes}_transformed3D.npy > test.txt
        """

}

// preparing to cluster the transformed data
transformedData.splitText()
                .filter {it.toString().size() > 1}
                .set {transformedData1}

transformedData1.unique()
                .map {it -> it - 'transformed3D.npy' - '\n'}
                .set {transformedData2}

//////////////////////////////////////////////////////////////////////////////////
// Cluster the transformed data according to the number of subgenomes specified

process cluster {
    cpus = { clust == 1 ? 2 : 1 }

input:
    val transformedData2
    each model from clusterModels

output:
    file 'test2.txt' into subgenomeFoldersRaw

script:
    if(clust == 1)
        """
        #!/bin/bash
        cd ${workingDir}
        polycracker cluster --file=${transformedData2}transformed3D.npy --cluster_method=${model} --n_subgenomes=${n_subgenomes} --metric=${metric} --n_neighbors=${n_neighbors} --weighted_nn=${weighted_nn} --grab_all=${grabAll} -mst ${mst}
        echo ${model}
        cd -
        echo ${model}${transformedData2}n${n_clusters} > test2.txt
        """
    else
        """
        echo ${model}${transformedData2}n${n_clusters} > test2.txt
        """

}

// Prepare subgenomes for extraction from cluster data
subgenomeFoldersRaw.splitText()
                .filter {it.toString().size() > 1}
                .set {subgenomeFolders}
subgenomeFolders.map { it -> it - '\n' }
                .set { subgenomeFoldersFinal }

//////////////////////////////////////////////////////////////////////////////////
// Extract Subgenomes based on clustering results

process subgenomeExtraction {
    cpus = { extract == 1 ? 4 : 1 }

input:
    val subgenomeFolder from subgenomeFoldersFinal

script:
    if(extract == 1)
        """
        #!/bin/bash
        cd ${workingDir}
        polycracker subgenomeExtraction --subgenome_folder=./analysisOutputs/${subgenomeFolder}/bootstrap_0 --original_subgenome_path=./analysisOutputs/${subgenomeFolder} --fasta_path=${fastaPath} --genome_name=${genomeSplitName} --original_genome=${genome2} --bb=1 --bootstrap=${bootstrap} --iteration=0 --kmer_length=${kmerLength} --run_final=0 --original=${originalStr} --blast_mem=${blastMemory} --kmer_low_count=${kmer_low_count} --diff_kmer_threshold=${diff_kmer_threshold} --unionbed_threshold=${unionbed_threshold} --diff_sample_rate=${diff_sample_rate} --default_kmercount_value=${default_kmercount_value} -pm ${perfect_mode} -sl ${k_search_length}
        cd -
        touch test.txt
        """
    else
        """
        touch test.txt
        """
}

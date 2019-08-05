#!/bin/bash
export _JAVA_OPTIONS='-Xms3G -Xmx5G'
nextflow run -process.echo true polycracker.nf -with-timeline timeline.html -with-dag flowchart.pdf

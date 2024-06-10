# ArraySeq

Processing and analysis of spatial transcriptomics datasets generated by Array-seq.
The Array-seq cature area size, makes it well-suited for multiple replicates, and high throughput experiments. The ArraySeq package is therefore equipt with functionalities for handling:
- Single tissue sections
- Multiples tissue sections
- Serial sections for 3D spatial transcriptomics
  - Sections spacing up to 100 µm have been validated. 
  
**Please cite:** *Insert Citation Here

## Schematic Workflow

![Workflow](https://github.com/chevrierlab/ArraySeq/assets/63480747/7288dd33-7147-4bc2-b46e-c84c15308bc4)

## Pre-processing

Following the generation of fastq files, we recommend using STARSolo (>=2.7.8a) for denulteplexing spatial barcodes and counting UMIs. The barcode.txt file can be downloaded from the **Insert folder here** folder. Other barcode files from custom-made array designs can be used in its place. Below is an example script:

```bash
export PATH=<Path to STAR (>=2.7.8a) static build folder>:$PATH

Fastq_1=<Path to Read1.fastq.gz file>
Fastq_2=<Path to Read2.fastq.gz file>
Sample_name=<Your_Sample_Name>

STAR \
  --runThreadN 24 \
  --genomeDir <Path to Genome Index Folder> \
  --soloType CB_UMI_Simple \
  --soloCBwhitelist <Path to n12_barcodes.txt> \
  --soloCBstart 1 \
  --soloCBlen 12 \
  --soloUMIstart 29 \
  --soloUMIlen 10 \
  --soloBarcodeReadLength 0 \
  --soloUMIdedup 1MM_CR \
  --soloCBmatchWLtype Exact \
  --soloFeatures GeneFull \
  --outFilterScoreMinOverLread 0 \
  --outFilterMatchNminOverLread 0 \
  --clip3pAdapterSeq AAAAAAAAAA \
  --clip3pAdapterMMp 0.1 \
  --readFilesIn $Fastq_2 $Fastq_1 \
  --readFilesCommand zcat \
  --soloOutFileNames ${Sample_name}_ST/ features.tsv barcodes.tsv matrix.mtx \
  --outFileNamePrefix ${Sample_name}_

gzip ${Sample_name}_ST/GeneFull/raw/*

rm ${Sample_name}_Aligned.out.sam
```

## Usage and Documentatation



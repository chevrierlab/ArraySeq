export PATH=/project2/nchevrier/software/STAR_2.7.10a/STAR-2.7.10a/bin/Linux_x86_64_static/:$PATH

Fastq_1=<Path to Read1 Fastq File>
Fastq_2=<Path to Read2 Fastq File>
barcodes=<Path to Barcode White List>

STAR \
  --runThreadN 24 \
  --genomeDir /project2/nchevrier/reference_data/ST_reference \
  --soloType CB_UMI_Simple \
  --soloCBwhitelist $barcodes \
  --soloCBstart 1 \
  --soloCBlen 18 \
  --soloUMIstart 35 \
  --soloUMIlen 10 \
  --soloBarcodeReadLength 0 \
  --soloUMIdedup 1MM_CR \
  --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts \
  --soloFeatures Gene \
  --clip3pAdapterSeq AAAAAAAAAA \
  --clip3pAdapterMMp 0.1 \
  --outFilterScoreMinOverLread 0 \
  --outFilterMatchNminOverLread 0 \
  --readFilesIn $Fastq_2 $Fastq_1 \
  --readFilesCommand zcat \
  --soloOutFileNames Solo.out/ features.tsv barcodes.tsv matrix.mtx


gzip Solo.out/Gene/raw/*

rm Aligned.out.sam

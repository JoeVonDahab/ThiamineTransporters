set -u
D=/home/joe/projects/TripleVar/data/external_raw/benchmarks
cd $D
head -c 300 /home/joe/projects/TripleVar/data/reference/SLC19A2.window.fa; echo
grep '>' /home/joe/projects/TripleVar/data/reference/*.fa
ls -la /home/joe/projects/TripleVar/data/reference/
cat /home/joe/projects/TripleVar/data/reference/README* 2>/dev/null | head -40
zcat AlphaMissense_hg38.tsv.gz | awk -F'\t' 'BEGIN{OFS="\t"} /^#CHROM/{print; next} /^#/{next} ($1=="chr1" && $2>=169461909 && $2<=169487970) || ($1=="chr2" && $2>=227682763 && $2<=227720028){print}' > AlphaMissense_hg38.SLC19A2_SLC19A3.tsv
wc -l AlphaMissense_hg38.SLC19A2_SLC19A3.tsv
cut -f1,6,7 AlphaMissense_hg38.SLC19A2_SLC19A3.tsv | sort | uniq -c

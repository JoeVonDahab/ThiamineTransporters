set -u
D=/home/joe/projects/TripleVar/data/external_raw/benchmarks
cd $D
unzip -p revel-v1.3_all_chromosomes.zip revel_with_transcript_ids | head -3
time unzip -p revel-v1.3_all_chromosomes.zip revel_with_transcript_ids | awk -F',' 'NR==1{print; next} ($1=="1" && $3!="." && $3>=169461909 && $3<=169487970) || ($1=="2" && $3!="." && $3>=227682763 && $3<=227720028) || ($3=="." && (($1=="1" && $2>=169330000 && $2<=169560000) || ($1=="2" && $2>=228440000 && $2<=228700000))) {print}' > revel-v1.3.SLC19A2_SLC19A3.csv
wc -l revel-v1.3.SLC19A2_SLC19A3.csv
awk -F',' 'NR>1{print $1, ($3=="."?"nogrch38":"ok"), $9}' revel-v1.3.SLC19A2_SLC19A3.csv | sort | uniq -c | sort -k1,1nr | head -30

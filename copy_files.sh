for i in {1..45}
do
	scp slurmq:/projects/mhadlich_segmentation/data/"$i"/log.txt data/"$i"_log.txt
	scp slurmq:/projects/mhadlich_segmentation/data/"$i"/usage.csv data/"$i"_usage.csv
done

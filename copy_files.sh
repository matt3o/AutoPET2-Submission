for i in {50..100}
do
	scp slurmq:/projects/mhadlich_segmentation/data/"$i"/log.txt data/"$i"_log.txt
	scp slurmq:/projects/mhadlich_segmentation/data/"$i"/usage.csv data/"$i"_usage.csv
done

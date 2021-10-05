raw_wavs_path='RUSLAN/'
dumpdir='RUSLAN_dump/'

cd $raw_wavs_path
for i in *.wav; 
do 
    sox "$i" -c 1 -b 16 -r 22050  "../$dumpdir/$i"; 
done

./prepare_mel.py $arg_opts --dumpdir $dumpdir

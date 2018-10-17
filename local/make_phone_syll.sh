#!/bin/bash
. cmd.sh
. path.sh

trans_file=
trans_table=
tgt_dir=

. parse_options.sh || exit 1;

tmpdir=$tgt_dir/tmp

mkdir -p  $tgt_dir $tmpdir

# convert ark back to trans.syll 
copy-int-vector scp:$trans_file ark,t:- | utils/int2sym.pl -f 2- \
	$trans_table > $tmpdir/trans.syll  || exit 1;

# convert trans.syll to trans.ph
cut -d " " -f2- $tmpdir/trans.syll | tr "_" " " > $tmpdir/ph.txt || exit 1;
cut -d " " -f1 $tmpdir/trans.syll > $tmpdir/ids.txt
paste -d " " $tmpdir/ids.txt  $tmpdir/ph.txt > $tmpdir/trans.ph

# get syllabes from training data
awk '{for(i=2; i<=NF; i++) print $i}'  $tmpdir/trans.syll |	tr " " "\n" | \
   	sort | uniq  > $tmpdir/syllables.txt || exit 1;

#(echo "SPN"; echo "NSN";) | cat -  $tmpdir/syllabes.txt | sort |  uniq  > $tmpdir/syllables.txt.1 || exit 1;
echo "<blk>" >> $tmpdir/syllables.txt
cat $tmpdir/syllables.txt | awk '{print $0 " " NR-1}' > $tgt_dir/trans.syll.txt || exit 1;

# get phones from training data

awk '{for(i=2; i<=NF; i++) print $i}'  $tmpdir/trans.syll | tr "_" " " | tr " " "\n" | \
   	sort | uniq  > $tmpdir/phones.txt || exit 1;

#(echo "SPN"; echo "NSN";) | cat $tmpdir/phones.txt | sort |  uniq > $tmpdir/phones.txt.1 || exit 1;
echo "<blk>" >> $tmpdir/phones.txt
cat $tmpdir/phones.txt |  awk '{print $0 " " NR-1}' > $tgt_dir/trans.ph.txt || exit 1;


# convert back to ark using different labels
echo "comvert back to ark using syllabes and phones"
utils/sym2int.pl -f 2- $tgt_dir/trans.ph.txt < $tmpdir/trans.ph | \
	copy-int-vector ark:- ark,scp:$tgt_dir/feats_ph.ark,$tgt_dir/feats_ph.scp || exit 1;

utils/sym2int.pl -f 2- $tgt_dir/trans.syll.txt < $tmpdir/trans.syll | \
	copy-int-vector ark:- ark,scp:$tgt_dir/feats_syll.ark,$tgt_dir/feats_syll.scp|| exit 1;

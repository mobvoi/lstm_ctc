# Copyright 2018 Mobvoi Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.

. ./path.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 <lang-dir> <lm-dir> "
  echo "e.g.: data/lang data/lm data/lm_tmp"
  exit 1
fi

langdir=$1
lmdir=$2

# Adapted from the kaldi-trunk librespeech script local/format_lms.sh
export LC_ALL=C
echo "Preparing language models for large const lm, may take some time ... "
for lm_suffix in tglarge fglarge ; do
    test=${langdir}_test_const_${lm_suffix}
	echo $test
    mkdir -p $test
	cp -r $langdir/* $test
	nlines=`cat $test/words.txt | wc -l`
	unk=`grep "<UNK>" $test/words.txt | awk '{print $2}'`
	echo "<s> $[$nlines]" >> $test/words.txt
	echo "</s> $[$nlines+1]" >> $test/words.txt

    bos=`grep "<s>" $test/words.txt | awk '{print $2}'`
    eos=`grep "</s>" $test/words.txt | awk '{print $2}'`
    if [[ -z $bos || -z $eos ]]; then
	   	echo "$0: <s> and </s> symbols are not in $test/words.txt"
        exit 1
    fi

    arpa_lm=$lmdir/lm_${lm_suffix}.arpa.gz
    arpa-to-const-arpa --bos-symbol=$bos \
		--eos-symbol=$eos --unk-symbol=$unk \
		"gunzip -c $arpa_lm | utils/map_arpa_lm.pl $test/words.txt|"  $test/G.carpa  || exit 1;


done


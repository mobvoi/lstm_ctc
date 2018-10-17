#!/bin/bash

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

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
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script compiles the ARPA-formatted language models into FSTs. Finally it composes the LM, lexicon
# and token FSTs together into the decoding graph. 

. ./path.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 <lang-dir> <lm-dir> <lm-dir-tmp> <test_dir>"
  echo "e.g.: data/lang data/lm data/lm_tmp data/test"
  exit 1
fi

langdir=$1
lm=$2
tmpdir=$3
test=$4
mkdir -p $tmpdir
mkdir -p $test


# Adapted from the kaldi-trunk librespeech script local/format_lms.sh
echo "Preparing language models for testing, may take some time ... "
    cp ${langdir}/words.txt $testdir || exit 1;

    echo "-----------------------------------------"
    echo "Working on " $lm_suffix;
    echo "-----------------------------------------"

    cat $lm | utils/find_arpa_oovs.pl $test/words.txt  > $tmpdir/oovs_${lm_suffix}.txt

    # grep -v '<s> <s>' because the LM seems to have some strange and useless
    # stuff in it with multiple <s>'s in the history.
      cat $lm | \
      grep -v '<s> <s>' | \
      grep -v '</s> <s>' | \
      grep -v '</s> </s>' | \
      arpa2fst - | fstprint | \
      utils/remove_oovs.pl $tmpdir/oovs_${lm_suffix}.txt | \
      utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
        --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
       fstrmepsilon | fstarcsort --sort_type=ilabel > $test/G.fst
    
    # Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
    # minimized.
    fsttablecompose ${langdir}/L.fst $test/G.fst | fstdeterminizestar --use-log=true | \
      fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;
    fsttablecompose ${langdir}/T.fst $tmpdir/LG.fst > $test/TLG.fst || exit 1;

echo "Composing decoding graph TLG.fst succeeded"
rm -r $tmpdir

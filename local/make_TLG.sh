#!/bin/bash

. path.sh
. cmd.sh

srcdir=
dir=

. parse_options.sh || exit 1

tmpdir=$dir/tmp

mkdir -p $dir $tmpdir

cat $srcdir/lexicon.txt |\
  utils/sym2int.pl -f 2- $srcdir/units.txt \
  > $tmpdir/lexicon_numbers.txt

cat $srcdir/lexicon.txt |\
  perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' \
  > $tmpdir/lexiconp.txt || exit 1;

ndisambig=$(utils/add_lex_disambig.pl $tmpdir/lexiconp.txt $tmpdir/lexiconp_disambig.txt)
ndisambig=$[$ndisambig+1];

( for n in $(seq 0 $ndisambig); do echo '#'$n; done ) > $tmpdir/disambig.list
cat $srcdir/units.txt | awk '{print $1}' > $tmpdir/units.list
(echo '<eps>'; echo '<blk>';) |\
  cat - $tmpdir/units.list $tmpdir/disambig.list |\
  awk '{print $1 " " (NR-1)}' \
  > $dir/tokens.txt

# Compile the tokens into FST
local/ctc_token_fst.py $dir/tokens.txt |\
  fstcompile \
    --isymbols=$dir/tokens.txt --osymbols=$dir/tokens.txt \
   --keep_isymbols=false --keep_osymbols=false |\
  fstarcsort --sort_type=olabel > $dir/T.fst || exit 1;

# Encode the words with indices. Will be used in lexicon and language model FST compiling. 
cat $tmpdir/lexiconp.txt |\
  awk '{print $1}' |\
  sort |\
  uniq |\
  awk '
  BEGIN {
    print "<eps> 0";
  } 
  {
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
  }' > $dir/words.txt || exit 1;

# Now compile the lexicon FST. Depending on the size of your lexicon, it may take some time. 
token_disambig_symbol=$(grep \#0 $dir/tokens.txt | awk '{print $2}')
word_disambig_symbol=$(grep \#0 $dir/words.txt | awk '{print $2}')

local/make_lexicon_fst.pl \
  --pron-probs $tmpdir/lexiconp_disambig.txt 0 "sil" '#'$ndisambig |\
  fstcompile \
    --isymbols=$dir/tokens.txt --osymbols=$dir/words.txt \
    --keep_isymbols=false --keep_osymbols=false |\
  fstaddselfloops \
    "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" |\
  fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;

cat $srcdir/lm.arpa.txt |\
  utils/find_arpa_oovs.pl $dir/words.txt \
  > $tmpdir/oovs.txt || exit 1

cat $srcdir/lm.arpa.txt |\
  /nfs/cfyeh/tools/kaldi/src/lmbin/arpa2fst \
    --disambig-symbol=#0 \
    --read-symbol-table=$dir/words.txt \
    - - |\
  fstarcsort --sort_type=ilabel \
    > $dir/G.fst || exit 1

echo "composing L.fst and G.fst -> $tmpdir/LG.fst"
fsttablecompose $dir/L.fst $dir/G.fst |\
  fstdeterminizestar --use-log=true |\
  fstminimizeencoded |\
  fstarcsort --sort_type=ilabel \
  > $tmpdir/LG.fst || exit 1

echo "composing T.fst LG.fst -> $dir/TLG.fst"
fsttablecompose $dir/T.fst $tmpdir/LG.fst \
  > $dir/TLG.fst || exit 1

exit 0

#!/bin/bash

. cmd.sh
. path.sh

################################################################################
# Set up variables.
################################################################################

tfrecords_scp=
nnet_config=
blank_index=
graphdir=
ans=
dir=

apply_softmax=false

cmd=run.pl
nj=16
acwt=0.9
min_active=200
max_active=7000 # max-active
beam=15.0       # beam used
lattice_beam=8.0
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes

scoring_opts="--min-acwt 1 --max-acwt 20"

echo
echo "$0 $@"  # Print the command line for logging
echo

. parse_options.sh || exit 1

[ -z "$tfrecords_scp" ] && echo -e "(ERROR) missing --tfrecords-scp\n" && exit 1
[ -z "$nnet_config" ] && echo -e "(ERROR) missing --nnet-config\n" && exit 1
[ -z "$graphdir" ] && echo -e "(ERROR) missing --graphdir\n" && exit 1
[ -z "$ans" ] && echo -e "(ERROR) missing --ans\n" && exit 1
[ -z "$dir" ] && echo -e "(ERROR) missing --dir\n" && exit 1


[ ! -e "$tfrecords_scp" ] && echo -e "(ERROR) $tfrecords_scp does not exist\n" && exit 1
[ ! -e "$nnet_config" ] && echo -e "(ERROR) $nnet_config does not exist\n" && exit 1
[ ! -e "$graphdir/TLG.fst" ] && echo -e "(ERROR) $graphdir/TLG.fst does not exist\n" && exit 1
[ ! -e "$graphdir/words.txt" ] && echo -e "(ERROR) $graphdir/words.txt does not exist\n" && exit 1
[ ! -e "$ans" ] && echo -e "(ERROR) $ans does not exist\n" && exit 1

################################################################################
# Get posterior from the neural network.
################################################################################

if [ -z "$nnet" ]; then
  srcdir=$(dirname $dir)
  nnet=$srcdir/$(cat $srcdir/final.nnet)
fi

mkdir -p $dir
if [ ! -e $dir/forward.done ]; then
  echo "[$(date +'%Y/%m/%d %H:%M:%S')] computing inference for posteriors"
  ( python bin/nnet-forward.py \
     --apply-softmax=$apply_softmax \
     $tfrecords_scp $nnet_config $nnet ark:- |\
     copy-feats ark:- ark,scp:$dir/post.ark,$dir/post.scp ) \
    2> $dir/forward.log || exit 1
  touch $dir/forward.done
else
  echo "[$(date +'%Y/%m/%d %H:%M:%S')] $dir/forward.done exists, skipping inference"
fi

if [ ! -e $dir/latgen.done ]; then
  echo "[$(date +'%Y/%m/%d %H:%M:%S')] generating lattices"
  for n in $(seq $nj); do
    utils/split_scp.pl -j $nj $[$n-1] $dir/post.scp $dir/post.$n.scp
  done

  num_targets=$(feat-to-dim scp:$dir/post.scp - 2> /dev/null)
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    copy-feats scp:$dir/post.JOB.scp ark:- \| \
    select-feats $[$num_targets-1],0-$[$num_targets-2] ark:- ark:- \| \
    latgen-faster \
      --max-active=$max_active \
      --max-mem=$max_mem \
      --beam=$beam \
      --lattice-beam=$lattice_beam \
      --acoustic-scale=$acwt \
      --allow-partial=true \
      --word-symbol-table=$graphdir/words.txt \
      $graphdir/TLG.fst ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1
  touch $dir/latgen.done
else
  echo "[$(date +'%Y/%m/%d %H:%M:%S')] $dir/latgen.done exists, skipping lattice generation"
fi

if [ ! -e $dir/score.done ]; then
  echo "[$(date +'%Y/%m/%d %H:%M:%S')] scoring"
  local/score.sh $scoring_opts $ans $graphdir $dir || exit 1;
  touch $dir/score.done
else
  echo "[$(date +'%Y/%m/%d %H:%M:%S')] $dir/score.done exists, skipping scoring"
fi

exit 0;

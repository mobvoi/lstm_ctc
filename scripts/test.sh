#!/bin/bash

. path.sh

################################################################################
# Set up variables.
################################################################################

tfrecords_scp=
nnet_config=
nnet=  # Optional, if blank then use the final.nnet in upper-level folder.
num_targets=
blank_index=
sequence=
ans=
dir=

apply_softmax=true
roc_x_min=0.00
roc_x_max=3.0
roc_y_min=0.00
roc_y_max=40.0

echo
echo "$0 $@"  # Print the command line for logging
echo

. parse_options.sh || exit 1

[ -z "$tfrecords_scp" ] && echo -e "(ERROR) missing --tfrecords-scp\n" && exit 1
[ -z "$nnet_config" ] && echo -e "(ERROR) missing --nnet-config\n" && exit 1
[ -z "$sequence" ] && echo -e "(ERROR) missing --sequence\n" && exit 1
[ -z "$ans" ] && echo -e "(ERROR) missing --ans\n" && exit 1
[ -z "$dir" ] && echo -e "(ERROR) missing --dir\n" && exit 1

[ ! -e "$tfrecords_scp" ] && echo -e "(ERROR) $tfrecords_scp does not exist\n" && exit 1
[ ! -e "$nnet_config" ] && echo -e "(ERROR) $nnet_config does not exist\n" && exit 1
[ ! -e "$ans" ] && echo -e "(ERROR) $ans does not exist\n" && exit 1

################################################################################
# Assign defaults to unspecified but neccesary variables.
################################################################################

[ -z "$blank_index" ] && blank_index=$[$num_targets - 1]

################################################################################
# Test neural network.
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

exit 0

if [ "$binary" == "hotword-detect-mapped" ]; then
  data=""
  for smooth_window in 1 2 4 8 16 32; do
    if [ ! -e $dir/$smooth_window.done ]; then
      echo "[$(date +'%Y/%m/%d %H:%M:%S')] detecting hotword with smoothing window = $smooth_window"
      cat $dir/post.ark |\
        hotwordbin/hotword-detect-mapped \
          --smooth-window=$smooth_window \
          "$sequence" ark:- ark,t:$dir/$smooth_window.score \
          2> $dir/$smooth_window.log || exit 1
      python bin/compute-roc.py \
        $dir/$smooth_window.score $ans $dir/$smooth_window.roc || exit 1
      touch $dir/$smooth_window.done
    else
      echo "[$(date +'%Y/%m/%d %H:%M:%S')] $dir/$smooth_window.done exists, skipping"
    fi
    [ "$data" != "" ] && data="$data "
    data="${data}$smooth_window.roc $dir/$smooth_window.roc"
  done
  
  png=$dir/roc.png
  python bin/draw-roc.py \
    --x-min=$roc_x_min --x-max=$roc_x_max \
    --y-min=$roc_y_min --y-max=$roc_y_max \
    $png $data
fi

if [ "$binary" == "hotword-graph-detect-mapped" ]; then
  if [ ! -e $dir/score.done ]; then
    cat $dir/post.ark |\
      hotwordbin/hotword-graph-detect-mapped \
        --num-targets=$num_targets \
        "$sequence" ark:- ark,t:$dir/score \
        2> $dir/score.log || exit 1
    touch $dir/score.done
  else
    echo "[$(date +'%Y/%m/%d %H:%M:%S')] $dir/score.done exists, skipping"
  fi
  if [ ! -e $dir/roc.done ]; then
    python bin/compute-roc.py \
      --step=0.0001 \
      $dir/score $ans $dir/roc || exit 1
    touch $dir/roc.done
  else
    echo "[$(date +'%Y/%m/%d %H:%M:%S')] $dir/roc.done exists, skipping"
  fi
fi

echo "[$(date +'%Y/%m/%d %H:%M:%S')] testing finished, the ROC curves are in $dir/roc.png"
echo

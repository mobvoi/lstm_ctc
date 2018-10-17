#!/bin/bash

. path.sh

################################################################################
# Set up variables.
################################################################################

feats_scp=
trans_scp=
cmvn_ark=
dir=

check_length=false
nj=8
cmd=run.pl

echo
echo "$0 $@"  # Print the command line for logging
echo

. parse_options.sh || exit 1

[ -z "$feats_scp" ] && echo -e "(ERROR) missing --feats-scp\n" && exit 1
[ -z "$cmvn_ark" ] && echo -e "(ERROR) missing --cmvn-ark\n" && exit 1
[ -z "$dir" ] && echo -e "(ERROR) missing --dir\n" && exit 1

[ ! -e "$feats_scp" ] && \
  echo -e "(ERROR) $feats_scp does not exist\n" && exit 1
[ ! -e "$cmvn_ark" ] && \
  echo -e "(ERROR) $cmvn_ark does not exist\n" && exit 1

[ ! -z "$trans_scp" ] && [ ! -e "$trans_scp" ] && \
  echo -e "(ERROR) $trans_scp does not exist\n" && exit 1

################################################################################
# Convert to TFRecords
################################################################################

mkdir -p $dir $dir/split${nj} $dir/log

echo "[$(date +'%Y/%m/%d %H:%M:%S')] generating TFRecords in $dir"

echo "splitting $nj jobs in $dir/split${nj}"
for n in $(seq $nj); do
  subdir=$dir/split${nj}/$n
  mkdir -p $subdir
  utils/split_scp.pl -j $nj $[$n-1] $feats_scp $subdir/feats.scp
  if [ ! -z "$trans_scp" ]; then
    cat $trans_scp |\
      utils/filter_scp.pl -f 1 $subdir/feats.scp \
      > $subdir/trans.scp
  fi
done

subdir=$(readlink -f $dir)/split${nj}/JOB
nnet_input="ark:cat $subdir/feats.scp |"
nnet_input="$nnet_input copy-feats scp:- ark:- |"
nnet_input="$nnet_input apply-cmvn --norm-means=true --norm-vars=true $cmvn_ark ark:- ark:- |"
[ ! -z "$trans_scp" ] && nnet_target="scp:$subdir/trans.scp"

echo "converting TFRecords"

$cmd JOB=1:$nj $dir/log/tfrecords.JOB.log \
  python bin/convert-to-tfrecords.py \
    --check-length=$check_length \
    ${nnet_target:+ --nnet-target="$nnet_target"} \
    "$nnet_input" $subdir $subdir/tfrecords.scp || exit 1

echo "creating list of all tfrecords in $dir/tfrecords.scp"
for n in $(seq $nj); do
  cat $dir/split${nj}/$n/tfrecords.scp
done | sort -k1,1 -u > $dir/tfrecords.scp

echo "[$(date +'%Y/%m/%d %H:%M:%S')] done"
echo

exit 0

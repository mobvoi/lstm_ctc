#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

stage=1
wsj0=/nfs2/yyshi/LDC/LDC93S6B
wsj1=/nfs2/yyshi/LDC/LDC94S13B
gpus="0"
num_layers=4
learn_rate=0.001
dropout_rate=0.9
left_context=1
right_context=1
subsample=3
num_projects=320
num_experts=72
moe_temp=10.0
nnet_type=blstm
use_decay=2
target_length_cutoff=2
prior_label_sm=0
uniform_label_sm=0
use_bn=false
num_neurons=320    # number of memory cells in every LSTM layer
batch_size=32
halving_factor=0.7

. utils/parse_options.sh

savedir=$1
# add check for IRSTLM prune-lm
if ! prune-lm > /dev/null 2>&1; then
    echo "Error: prune-lm (part of IRSTLM) is not in path"
    echo "Make sure that you run tools/extras/install_irstlm.sh in the main Eesen directory;"
    echo " this is no longer installed by default."
    exit 1
fi
# Specify network structure and generate the network topology
  input_dim=120   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
  optimizer="adam"
  sort_by_len=true
  batch_threads=8
  report_interval=1


  export CUDA_VISIBLE_DEVICES=$gpus


  dir=exp/${nnet_type}_proj_${num_layers}_${num_neurons}_${num_projects}_${learn_rate}_l${left_context}r${right_context}_d${dropout_rate}_ex${num_experts}_moet${moe_temp}_usm${uniform_label_sm}_psm${prior_label_sm}_bs${batch_size}_hf${halving_factor}
  mkdir -p $dir
  #hostname=$(hostname)
  #case $hostname in 
  #    Red-*)
  #  	  tfdata=$PWD/data/tfrecord
  #  	  ;;
  #    mobvoi-*)
  #  	  tfdata=/cache/yyshi/tfrecord/wsj_filter_short
  #  	  wsj0=/export/data/LDC/LDC93S6B
  #  	  wsj1=/export/data/LDC/LDC94S13B
  #  	  ;;
  #esac
  tfdata=$PWD/data/tfrecord

## Setup up features
  norm_vars=true
  add_deltas=true
  echo $norm_vars > $dir/norm_vars  # output feature configs which will be used in decoding
  echo $add_deltas > $dir/add_deltas





if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "             Data Preparation and FST Construction                 "
  echo =====================================================================
  # Use the same datap prepatation script from Kaldi
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

  # Construct the phoneme-based lexicon from the CMU dict
  local/wsj_prepare_phn_dict.sh || exit 1;

  # Compile the lexicon and token FSTs
  utils/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

  # Compile the language-model FST and the final decoding graph TLG.fst
  local/wsj_decode_graph.sh data/lang_phn || exit 1;
fi

if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================
  # Split the whole training data into training (95%) and cross-validation (5%) sets
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_si284 data/train_tr95 data/train_cv05 || exit 1

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  for set in train_tr95 train_cv05; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 8 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done

  for set in test_dev93 test_eval92; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 8 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done
fi

if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                    TFRecords Generation                       "
  echo =====================================================================

  datadir=$tfdata
  mkdir -p $datadir
  mkdir -p $datadir/data
  
  # Label sequences; simply convert words into their label indices
  # In tensorflow, the <blk> index is n-1
  if [ ! -e $datadir/label.tr.scp ] || [ ! -e $datadir/label.tr.ark ]; then
     utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_tr95/text "<UNK>" | \
		 awk -v s=1 '{printf $1 " "; for(i=2;i<=NF;i++)printf($i-s)" "};{print FS}' | \
   copy-int-vector ark:- ark,scp:$datadir/label.tr.ark,$datadir/label.tr.scp || exit 1;
  fi

  if [ ! -e $datadir/label.cv.scp ] || [ ! -e $datadir/label.cv.ark ]; then
     utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_cv05/text "<UNK>" | \
		 awk -v s=1 '{printf $1 " "; for(i=2;i<=NF;i++)printf($i-s)" "};{print FS}' | \
   copy-int-vector ark:- ark,scp:$datadir/label.cv.ark,$datadir/label.cv.scp || exit 1;
  fi

##
  
  if $sort_by_len; then
    feat-to-len scp:data/train_tr95/feats.scp ark,t:- | awk '{print $2}' > $dir/len.tmp || exit 1;
    paste -d " " data/train_tr95/feats.scp $dir/len.tmp | sort -k3 -n - | awk -v m=$min_len '{ if ($3 >= m) {print $1 " " $2} }' > $dir/train.scp || exit 1;
    feat-to-len scp:data/train_cv05/feats.scp ark,t:- | awk '{print $2}' > $dir/len.tmp || exit 1;
    paste -d " " data/train_cv05/feats.scp $dir/len.tmp | sort -k3 -n - | awk '{print $1 " " $2}' > $dir/cv.scp || exit 1;
    rm -f $dir/len.tmp
	
	feats_tr="cat $dir/train.scp |"
    feats_cv="cat $dir/cv.scp |"
    feats_tr="$feats_tr copy-feats scp:- ark:- |"
    feats_cv="$feats_cv copy-feats scp:- ark:- |"
  else
	feats_tr="cat data/train_tr95/feats.scp | utils/shuffle_list.pl --srand ${seed:-777}|"
    feats_cv="cat data/train_cv05/feats.scp | utils/shuffle_list.pl --srand ${seed:-777}|"
    feats_tr="$feats_tr copy-feats scp:- ark:- |"
    feats_cv="$feats_cv copy-feats scp:- ark:- |"
  fi


  feats_tr="$feats_tr apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:data/train_tr95/utt2spk scp:data/train_tr95/cmvn.scp  ark:- ark:- |"
  feats_cv="$feats_cv apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:data/train_cv05/utt2spk scp:data/train_cv05/cmvn.scp  ark:- ark:- |"

  if $add_deltas; then
    feats_tr="$feats_tr add-deltas ark:- ark:- |"
    feats_cv="$feats_cv add-deltas ark:- ark:- |"
  fi
  ## End of feature setup


  if [ ! -e $datadir/tfrecords.tr.scp ]; then
    echo "[$(date +'%Y/%m/%d %H:%M:%S')] tfrecords.tr.scp"
	nnet_input="ark:$feats_tr"
	nnet_target="scp:$datadir/label.tr.scp"  
	echo $nnet_input
    echo ${nnet_target:+ --nnet-target="$nnet_target"}
    python bin/convert-to-tfrecords.py \
		${nnet_target:+ --nnet-target="$nnet_target"} \
		--check-length=false --target-length-cutoff=$target_length_cutoff \
	    "$nnet_input"  $datadir/data $datadir/tfrecords.tr.scp || exit 1
  fi

  if [ ! -e $datadir/tfrecords.cv.scp ]; then
    echo "[$(date +'%Y/%m/%d %H:%M:%S')] tfrecords.cv.scp"
    nnet_input="ark:$feats_cv"
    nnet_target="scp:$datadir/label.cv.scp"  
    python bin/convert-to-tfrecords.py \
    	${nnet_target:+ --nnet-target="$nnet_target"} \
    	--check-length=false \
    	"$nnet_input"  $datadir/data $datadir/tfrecords.cv.scp || exit 1
  fi
fi

num_targets=`cat data/local/dict_phn/units.txt | wc -l`; num_targets=$[$num_targets+1]; # the number of targets 
                                                         # equals [the number of labels] + 1 (the blank)
 
  # Compute the occurrence counts of labels in the label sequences. These counts will be used to derive prior probabilities of
  # the labels.

  if [ ! -e $dir/label.counts ]; then
     utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_tr95/text "<UNK>" | gzip -c - > $dir/labels.tr.gz 
	 gunzip -c $dir/labels.tr.gz | awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' | \
  analyze-counts --verbose=1 --binary=false ark:- $dir/label.counts >& $dir/compute_label_counts.log || exit 1

 fi
prior_label_path=$dir/label.counts

if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                        Network Training                           "
  echo =====================================================================
 
if [ "$use_decay" == "1" ] ; then
	train_script=scripts/decay_train.sh
elif [ "$use_decay" == "2" ] ; then
	train_script=scripts/train_oplr.sh
else
	train_script=scripts/train.sh
fi



  
  nnet_config=$dir/nnet.config
  (echo "nnet_type = $nnet_type"
   echo "input_dim = $input_dim"
   echo "left_context = $left_context"
   echo "right_context = $right_context"
   echo "subsample = $subsample"
   echo "num_layers = $num_layers"
   echo "num_neurons = $num_neurons"
   echo "num_projects = $num_projects"
   echo "num_targets = $num_targets"
   echo "use_peepholes = true"
   echo "use_bn = $use_bn"
   echo "dropout_rate = $dropout_rate"
   echo "num_experts = $num_experts"
   echo "moe_temp = $moe_temp"
   echo "uniform_label_sm = $uniform_label_sm"
   echo "prior_label_sm = $prior_label_sm"
   echo "prior_label_path = $prior_label_path"
   echo "seed = 777") > $nnet_config

  $train_script \
    --objective "ctc" \
    --report-interval $report_interval \
    --batch-size $batch_size \
    --batch-threads $batch_threads \
    --tr-tfrecords-scp $tfdata/tfrecords.tr.scp \
    --cv-tfrecords-scp $tfdata/tfrecords.cv.scp \
    --nnet-config $nnet_config \
    --learn-rate $learn_rate \
    --optimizer $optimizer \
	--cv_goal loss \
	--halving_factor $halving_factor \
	--num_targets $num_targets \
	--decode_graph_dir data/lang_phn_test_tgpr \
	--decode_data_dir data/test_eval92 \
	--decode_name decode_eval92 \
    --dir $dir > $dir/train_log  || exit 1

fi

if [ $stage -le 5 ]; then

  echo =====================================================================
  echo "                            Decoding                               "
  echo =====================================================================
  # Config for the basic decoding: --beam 30.0 --max-active 5000 --acoustic-scales "0.7 0.8 0.9"
 


  for lm_suffix in tgpr tg; do
    scripts/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 8 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9  --ntargets $num_targets \
      data/lang_phn_test_${lm_suffix} data/test_dev93 $dir/decode_dev93_${lm_suffix} || exit 1;
    scripts/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 8 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9 --ntargets $num_targets \
      data/lang_phn_test_${lm_suffix} data/test_eval92 $dir/decode_eval92_${lm_suffix} || exit 1;
  done

fi

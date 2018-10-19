# Adapted from Kaldi librispeech and Eesen WSJ recipes by Jayadev Billa (2017)

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
[ -f path.sh ] && . ./path.sh;

stage=1
#libri=/export/data/en-asr-data/OpenSLR/LibriSpeech
libri=/nfs2/yyshi/OpenSLR/LibriSpeech
num_layers=4
learn_rate=0.0004
dropout_rate=0.9
left_context=1
right_context=1
subsample=3
num_projects=320
num_experts=44
moe_temp=20.0
nnet_type=blstm
batch_size=64
use_decay=2
target_length_cutoff=2
prior_label_sm=0
uniform_label_sm=0
use_bn=false
num_neurons=320    # number of memory cells in every LSTM layer
gpus=2
. utils/parse_options.sh

# Specify network structure and generate the network topology
  input_dim=120   # dimension of the input features; we will use 40-dimensional fbanks with deltas and double deltas
  optimizer="adam"
  sort_by_len=true
  batch_threads=8
  report_interval=1


export CUDA_VISIBLE_DEVICES=$gpus


dir=exp/libri_${nnet_type}_proj_${num_layers}_${num_neurons}_${num_projects}_${learn_rate}_l${left_context}r${right_context}_d${dropout_rate}_ex${num_experts}_moet${moe_temp}_bn${use_bn}_ud${use_decay}_usm${uniform_label_sm}_psm${prior_label_sm}
  mkdir -p $dir
 # hostname=$(hostname)
 # case $hostname in 
 #     Red-*)
 #   	  tfdata=$PWD/data/tfrecord
 #   	  ;;
 #     mobvoi-*)
 #   	  tfdata=/cache/yyshi/tfrecord/libri
 #   	  libri=/export/data/en-asr-data/OpenSLR/LibriSpeech
 #   	  ;;
 # esac
 tfdata=$PWD/data/tfrecord

## Setup up features
  norm_vars=true
  add_deltas=true
  echo $norm_vars > $dir/norm_vars  # output feature configs which will be used in decoding
  echo $add_deltas > $dir/add_deltas




data=data_libri
lm_data=$data/lm #data/local/lm
lm_tmp=$data/lm_tmp
dict_dir=$data/dict #data/local/dict
lang_dir=$data/lang #data/lang
feats_tmpdir=./tmp # this should ideally be a tmp dir local to the machine.
train_dir=$exp_base/train_lstm   # working directory

dict_name=librispeech_phn_reduced_dict.txt
dict_type="char"
fb_conf=$dir/fbconf

# create directories and copy relevant files
mkdir -p $data/{lm,lm_tmp,dict,lang}
cp conf/$dict_name $lm_data
cp conf/fbconf-{8,10,11} $dir

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

echo =====================================================================
echo "Started run @ ", `date`
echo =====================================================================

if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "             Data Preparation                                      "
  echo =====================================================================

  # download the 100hr training data and test sets.
  #for part in dev-clean test-clean dev-other test-other train-clean-100; do
  #    local/download_and_untar.sh $data $data_url $part || exit 1;
  #done

  # download the LM resources
  local/download_lm.sh $lm_url $lm_data || exit 1;

  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
      # use underscore-separated names in data directories.
      local/libri_data_prep.sh ${libri}/$part $data/$(echo $part | sed s/-/_/g) || exit 1;
  done
fi


if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                 Prepare dictionary and FST                    "
  echo =====================================================================

  ## See Kaldi librispeech recipe for additional information/context

  # Normally dict is in $lm_data but for this sequence of experiments well provide the dict
  # in $exp_base
  local/ls_prepare_phoneme_dict.sh $lm_data $dict_dir $dict_name || exit 1;

  # Compile the lexicon and token FSTs
  # usage: utils/ctc_compile_dict_token.sh <dict-src-dir> <tmp-dir> <lang-dir>"
  utils/ctc_compile_dict_token.sh --dict-type $dict_type --space-char "<SPACE>" \
    $dict_dir $lang_dir/tmp $lang_dir || exit 1;

  # Compile the language-model FST and the final decoding graph TLG.fst
  local/ls_decode_graph.sh $lang_dir $lm_data $lm_tmp/tmp || exit 1;
fi


if [ $stage -le 3 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================
  # combin train_clean_100, train_clean_360 and train_other_500
  mkdir -p $data/train
  for name in spk2gender  spk2utt utt2spk text  wav.scp ; do
	 cat $data/train_clean_100/$name $data/train_clean_360/$name $data/train_other_500/$name > $data/train/$name
  done
  
  # Split the whole training data into training (95%) and cross-validation (5%) sets
  # utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_si284 data/train_tr95 data/train_cv05 || exit 1
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 $data/train  $data/train_tr95 $data/train_cv05 || exit 1

  export  LC_ALL=C
  for set in train_tr95 train_cv05 dev_clean test_clean dev_other test_other ; do
  for name in utt2spk spk2utt  text wav.scp spk2gender ; do
  cat $data/$set/$name | sort -k1 > $data/$set/$name.b
  mv $data/$set/$name.b $data/$set/$name

  done
  done



  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  for set in train_tr95  train_cv05 ; do
  steps/make_fbank.sh --cmd "$train_cmd" --nj 14 --fbank-config ${fb_conf}-10 $data/$set $data/make_fbank/$set $data/$fbankdir || exit 1;
  utils/fix_data_dir.sh $data/$set || exit;
  steps/compute_cmvn_stats.sh $data/$set $data/make_fbank/$set $data/$fbankdir || exit 1;
  done

  for set in  dev_clean test_clean dev_other test_other ; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 14 --fbank-config ${fb_conf}-10  $data/$set $data/make_fbank/$set $data/$fbankdir || exit 1;
    utils/fix_data_dir.sh $data/$set || exit;
    steps/compute_cmvn_stats.sh $data/$set $data/make_fbank/$set $data/$fbankdir || exit 1;
  done
fi


if [ $stage -le 4 ]; then
  echo =====================================================================
  echo "                    TFRecords Generation                       "
  echo =====================================================================

  datadir=$tfdata
  mkdir -p $datadir
  mkdir -p $datadir/data
  
  # Label sequences; simply convert words into their label indices
  # In tensorflow, the <blk> index is n-1
  if [ ! -e $datadir/label.tr.scp ] || [ ! -e $datadir/label.tr.ark ]; then
     utils/prep_ctc_trans.py $lang_dir/lexicon_numbers.txt $data/train_tr95/text "<UNK>" "<SPACE>" | \
		 awk -v s=1 '{printf $1 " "; for(i=2;i<=NF;i++)printf($i-s)" "};{print FS}' | \
   copy-int-vector ark:- ark,scp:$datadir/label.tr.ark,$datadir/label.tr.scp || exit 1;
  fi

  if [ ! -e $datadir/label.cv.scp ] || [ ! -e $datadir/label.cv.ark ]; then
     utils/prep_ctc_trans.py $lang_dir/lexicon_numbers.txt $data/train_cv05/text "<UNK>" "<SPACE>"  | \
		 awk -v s=1 '{printf $1 " "; for(i=2;i<=NF;i++)printf($i-s)" "};{print FS}' | \
   copy-int-vector ark:- ark,scp:$datadir/label.cv.ark,$datadir/label.cv.scp || exit 1;
  fi

##
  
  if $sort_by_len; then
    feat-to-len scp:$data/train_tr95/feats.scp ark,t:- | awk '{print $2}' > $dir/len.tmp || exit 1;
    paste -d " " $data/train_tr95/feats.scp $dir/len.tmp | sort -k3 -n - | awk -v m=$min_len '{ if ($3 >= m) {print $1 " " $2} }' > $dir/train.scp || exit 1;
    feat-to-len scp:$data/train_cv05/feats.scp ark,t:- | awk '{print $2}' > $dir/len.tmp || exit 1;
    paste -d " " $data/train_cv05/feats.scp $dir/len.tmp | sort -k3 -n - | awk '{print $1 " " $2}' > $dir/cv.scp || exit 1;
    rm -f $dir/len.tmp
	
	feats_tr="cat $dir/train.scp |"
    feats_cv="cat $dir/cv.scp |"
    feats_tr="$feats_tr copy-feats scp:- ark:- |"
    feats_cv="$feats_cv copy-feats scp:- ark:- |"
  else
	feats_tr="cat $data/train_tr95/feats.scp | utils/shuffle_list.pl --srand ${seed:-777}|"
    feats_cv="cat $data/train_cv05/feats.scp | utils/shuffle_list.pl --srand ${seed:-777}|"
    feats_tr="$feats_tr copy-feats scp:- ark:- |"
    feats_cv="$feats_cv copy-feats scp:- ark:- |"
  fi


  feats_tr="$feats_tr apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/train_tr95/utt2spk scp:$data/train_tr95/cmvn.scp  ark:- ark:- |"
  feats_cv="$feats_cv apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/train_cv05/utt2spk scp:$data/train_cv05/cmvn.scp  ark:- ark:- |"

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

num_targets=`cat $data/dict/units.txt | wc -l`; num_targets=$[$num_targets+1]; # the number of targets 
                                                         # equals [the number of labels] + 1 (the blank)
 
  # Compute the occurrence counts of labels in the label sequences. These counts will be used to derive prior probabilities of
  # the labels.

  if [ ! -e $dir/label.counts ]; then
     utils/prep_ctc_trans.py $lang_dir/lexicon_numbers.txt $data/train_tr95/text "<UNK>" "<SPACE>" | gzip -c - > $dir/labels.tr.gz 
	 gunzip -c $dir/labels.tr.gz | awk '{line=$0; gsub(" "," 0 ",line); print line " 0";}' | \
  analyze-counts --verbose=1 --binary=false ark:- $dir/label.counts >& $dir/compute_label_counts.log || exit 1

 fi

 prior_label_path=$dir/labels.counts




if [ $stage -le 5 ]; then
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
   echo "uniform_label_sm = $uniform_label_sm"
   echo "prior_label_sm =  $prior_label_sm"
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
	--num_targets $num_targets \
	--decode_graph_dir $data/lang_test_tgmed \
	--decode_data_dir $data/dev_clean \
	--decode_name decode_dev_clearn \
    --dir $dir > $dir/train_log  || exit 1

fi

if [ $stage -le 6 ]; then
  echo =====================================================================
  echo "                            Decoding                               "
  echo =====================================================================

  # Decoding with the librispeech dict
  for test in test_clean test_other dev_clean dev_other; do
      for lm_suffix in tgsmall tgmed; do
          scripts/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 8 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9  --ntargets $num_targets \
      $data/lang_test_${lm_suffix} $data/$test $dir/decode_${test}_${lm_suffix} || exit 1;
      done
  done

fi

#!/bin/bash

. path.sh

################################################################################
# Set up variables.
################################################################################

tr_tfrecords_scp=
cv_tfrecords_scp=
nnet_config=
srcdir= # optional
dir=

objective="xent"

left_context=15
right_context=5

optimizer="momentum"
max_iter=50
min_iters=50 # keep training, disable weight rejection, start learn-rate halving as usual
keep_lr_iters=0 # fix learning rate for N initial epochs, disable weight rejection
learn_rate=0.008
start_halving_impr=0.001
end_halving_impr=0.0001
stop_halving_impr=0.01
halving_factor=0.5
min_learning_rate=0.00001
shuffle=false
seed=777

batch_size=256
max_batch_size=512
batch_threads=8
report_interval=100
cv_goal=eval
num_targets=72
decode_graph_dir="data/lang_phn_test_tgpr"
decode_data_dir=data/test_eval92
decode_name=decode_eval92
echo
echo "$0 $@"  # Print the command line for logging
echo

. parse_options.sh || exit 1

[ -z "$tr_tfrecords_scp" ] && echo -e "(ERROR) missing --tr-tfrecords-scp\n" && exit 1
[ -z "$cv_tfrecords_scp" ] && echo -e "(ERROR) missing --cv-tfrecords-scp\n" && exit 1
[ -z "$srcdir" ] && [ -z "$nnet_config" ] && \
  echo -e "(ERROR) missing --nnet-config or --srcdir\n" && exit 1
[ -z "$dir" ] && echo -e "(ERROR) missing --dir\n" && exit 1

[ ! -z "$srcdir" ] && [ -z "$nnet_config" ] && nnet_config="$srcdir/nnet.config"

[ ! -e "$tr_tfrecords_scp" ] && echo -e "(ERROR) $tr_tfrecords_scp does not exist\n" && exit 1
[ ! -e "$cv_tfrecords_scp" ] && echo -e "(ERROR) $cv_tfrecords_scp does not exist\n" && exit 1
[ ! -e "$nnet_config" ] && echo -e "(ERROR) $nnet_config does not exist\n" && exit 1

mkdir -p $dir

([ ! -z "$srcdir" ] || \
 [ "$(readlink -f $nnet_config)" != "$(readlink -f $dir/nnet.config)" ]) && \
(cp $nnet_config $dir/nnet.config || exit 1)
nnet_config=$dir/nnet.config

################################################################################
# Iteration 0 operations.
################################################################################

iter=0
echo "[$(date +'%Y/%m/%d %H:%M:%S')] iteration $iter"
if [ ! -z "$srcdir" ]; then
  nnet_best="$srcdir/$(cat $srcdir/final.nnet)"
  if [ ! -e $dir/nnet.${iter}.done ]; then
    python  bin/nnet-validate.py \
      --objective=$objective \
      --evaluate=true \
      --batch-size $batch_size \
      --batch-threads $batch_threads \
      --report-interval=$report_interval \
      $cv_tfrecords_scp $nnet_config $nnet_best \
      2> $dir/nnet.${iter}.cv.log || exit 1
    cv_loss=$(grep "^INFO:tensorflow:cv_loss" $dir/nnet.${iter}.cv.log | awk '{print $NF}')
    cv_eval=$(grep "^INFO:tensorflow:cv_eval" $dir/nnet.${iter}.cv.log | awk '{print $NF}')
    (echo "cv_loss $cv_loss"; echo "cv_eval $cv_eval") > $dir/nnet.${iter}.done
  fi
else
  nnet_best=$dir/nnet.${iter}
  if [ ! -e $dir/nnet.${iter}.done ]; then
    python  bin/nnet-init.py \
      --objective=$objective \
      --evaluate=true \
      --batch-size=$batch_size \
      --batch-threads=$batch_threads \
      --report-interval=$report_interval \
      $cv_tfrecords_scp $nnet_config $nnet_best \
      2> $dir/nnet.${iter}.cv.log || exit 1
    cv_loss=$(grep "^INFO:tensorflow:cv_loss" $dir/nnet.${iter}.cv.log | awk '{print $NF}')
    cv_eval=$(grep "^INFO:tensorflow:cv_eval" $dir/nnet.${iter}.cv.log | awk '{print $NF}')
    (echo "cv_loss $cv_loss"; echo "cv_eval $cv_eval") > $dir/nnet.${iter}.done
  fi
fi
cv_loss_best=$(grep "^cv_loss" $dir/nnet.${iter}.done | awk '{print $NF}')
cv_eval_best=$(grep "^cv_eval" $dir/nnet.${iter}.done | awk '{print $NF}')

if [ "$cv_goal" == "loss" ]; then
  cv_goal_best=$cv_loss_best
elif [ "$cv_goal" == "eval" ]; then
  cv_goal_best=$cv_eval_best
else
  echo "ERROR: unsupported cv_goal = $cv_goal" && exit 1
fi
echo "cv_goal_best = $cv_goal_best"

################################################################################
# Train neural network.
################################################################################

halving=0
for iter in $(seq $max_iter); do
  prev_iter=$[$iter-1]
  nnet_in=$nnet_best
  nnet_out=$dir/nnet.$iter

  echo
  echo "[$(date +'%Y/%m/%d %H:%M:%S')] iteration $iter" 
  tr_loss=
  cv_loss=
  if [ ! -e $dir/nnet.${iter}.done ]; then
    echo "training with learn_rate = $learn_rate"
    echo "nnet_in = $nnet_in"
    echo "nnet_out = $nnet_out"
    python bin/nnet-train.py \
      --objective=$objective \
      --learn-rate=$learn_rate \
      --optimizer=$optimizer \
      --seed=$iter \
      --shuffle=$shuffle \
      --batch-size $batch_size \
      --batch-threads $batch_threads \
      --report-interval=$report_interval \
      $tr_tfrecords_scp $nnet_config $nnet_in $nnet_out \
      2> $dir/nnet.${iter}.tr.log 
    tr_loss=$(grep "^INFO:tensorflow:tr_loss" $dir/nnet.${iter}.tr.log | awk '{print $NF}')
    [ "$tr_loss" == "nan" ] && echo "(ERROR) tr_loss = $tr_loss"
	 
	if [ "$tr_loss" == "nan" ]; then
		echo "ERROR tr_loss = $tr_loss"
		echo "reduce learn rate and re-train"
        learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
		echo "training with learn_rate = $learn_rate"
        echo "nnet_in = $nnet_in"
        echo "nnet_out = $nnet_out"
        python bin/nnet-train.py \
          --objective=$objective \
          --learn-rate=$learn_rate \
          --optimizer=$optimizer \
          --seed=$iter \
          --shuffle=$shuffle \
          --batch-size $batch_size \
          --batch-threads $batch_threads \
          --report-interval=$report_interval \
          $tr_tfrecords_scp $nnet_config $nnet_in $nnet_out \
          2> $dir/nnet.${iter}.tr.log 
        tr_loss=$(grep "^INFO:tensorflow:tr_loss" $dir/nnet.${iter}.tr.log | awk '{print $NF}')
        [ "$tr_loss" == "nan" ] && echo "(ERROR) tr_loss = $tr_loss" && exit 1
	 fi


    python bin/nnet-validate.py \
      --objective=$objective \
      --evaluate=true \
      --batch-size $batch_size \
      --batch-threads $batch_threads \
      --report-interval=$report_interval \
      $cv_tfrecords_scp $nnet_config $nnet_out \
      2> $dir/nnet.${iter}.cv.log || exit 1
    cv_loss=$(grep "^INFO:tensorflow:cv_loss" $dir/nnet.${iter}.cv.log | awk '{print $NF}')
    cv_eval=$(grep "^INFO:tensorflow:cv_eval" $dir/nnet.${iter}.cv.log | awk '{print $NF}')
    [ "$cv_loss" == "nan" ] && echo "(ERROR) cv_loss = $cv_loss" && exit 1
    [ "$cv_eval" == "nan" ] && echo "(ERROR) cv_eval = $cv_eval" && exit 1

    (echo "tr_loss $tr_loss"; echo "cv_loss $cv_loss"; echo "cv_eval $cv_eval") \
      > $dir/nnet.${iter}.done
    # put the decoding into background
	echo "nnet.${iter}" > $dir/final.nnet 
	 scripts/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 8 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9  --ntargets $num_targets \
      $decode_graph_dir  $decode_data_dir $dir/${decode_name}_${iter} &>/dev/null &
	 process=$!

  else
    echo "$dir/nnet.${iter}.done exists, skipping this iteration"
    tr_loss=$(grep "^tr_loss" $dir/nnet.${iter}.done | awk '{print $NF}')
    cv_loss=$(grep "^cv_loss" $dir/nnet.${iter}.done | awk '{print $NF}')
    cv_eval=$(grep "^cv_eval" $dir/nnet.${iter}.done | awk '{print $NF}')
  fi
  echo "tr_loss = $tr_loss cv_loss = $cv_loss cv_eval = $cv_eval"

  if [ "$cv_goal" == "loss" ]; then
    cv_goal_val=$cv_loss
  elif [ "$cv_goal" == "eval" ]; then
    cv_goal_val=$cv_eval
  else
    echo "ERROR: unsupported cv_goal = $cv_goal" && exit 1
  fi

  rel_impr=$(awk "BEGIN{print(($cv_goal_best - $cv_goal_val) / $cv_goal_best);}")

  echo "cv_goal_val = $cv_goal_val cv_goal_best = $cv_goal_best relative improvement = $rel_impr"

  # accept or reject?
  if [ 1 == $(awk "BEGIN{print($cv_goal_val < $cv_goal_best ? 1:0);}") ]; then
    # accepting: the loss was better
    nnet_best=$nnet_out
    cv_eval_best=$cv_eval
    cv_loss_best=$cv_loss
    cv_goal_best=$cv_goal_val
    echo "nnet accepted ($(basename $nnet_best))"
  else
    # rejecting,
    echo "nnet rejected ($(basename $nnet_out))"
  fi

  # continue with original learn-rate,
  [ $iter -le $keep_lr_iters ] && continue

  # stopping criterion
  if [ 1 == $halving -a 1 == $(awk "BEGIN{print($rel_impr < $end_halving_impr ? 1:0);}") ]; then
    if [ $iter -le $min_iters ]; then
      echo "supposed to finish, but we continue as min_iters = $min_iters"
	  echo -n "halving learning rate from $learn_rate to "
	  learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
	  learn_rate=$(awk "BEGIN{print($learn_rate < $min_learning_rate ? $min_learning_rate : $learn_rate)}")
      echo "$learn_rate"
      continue
    fi
    echo "finished, too small rel. improvement $rel_impr < $end_halving_impr"
    break
  fi

  # start learning-rate fade-out when improvement is low,
  if [ 0 == $halving -a 1 == $(awk "BEGIN{print($rel_impr < $start_halving_impr ? 1:0);}") ]; then
    echo "start halving learning rate, small rel. improvement $rel_impr < $start_halving_impr"
    halving=1
  fi

  if [ 1 == $(awk "BEGIN{print($rel_impr > $stop_halving_impr ? 1:0);}") ]; then
	  echo "stop halving learning rate, big rel. improvement $rel_impr > $stop_halving_impr"
	  halving=0
  fi

  # reduce the learning-rate,
  if [ 1 == $halving ]; then
    echo -n "halving learning rate from $learn_rate to "
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
	learn_rate=$(awk "BEGIN{print($learn_rate < $min_learning_rate ? $min_learning_rate : $learn_rate)}")
    echo "$learn_rate"
  fi
done

echo "$(basename $nnet_best)" > $dir/final.nnet
echo "[$(date +'%Y/%m/%d %H:%M:%S')] training finished, the final model is $dir/$(cat $dir/final.nnet)"
echo

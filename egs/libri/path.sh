#!/bin/bash

KALDI_ROOT=../../../kaldi
EESEN_ROOT=../../../eesen
export KALDI_ROOT=$KALDI_ROOT
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
export PYTHONPATH=$PYTHONPATH:$PWD

export PATH=$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/extras/irstlm/bin/:$PWD/bin:$PWD/utils:$KALDI_ROOT/tools/openfst/bin:$PWD:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/extras/irstlm/bin/:$PWD:$PATH
[ ! -e steps ] && cp -r  $KALDI_ROOT/egs/wsj/s5/steps ./ && cp $EESEN_ROOT/asr_egs/wsj/steps/* ./steps/ && cp $EESEN_ROOT/asr_egs/librispeech/steps/* ./steps/
[ ! -e scripts ] && ln -sf ../../scripts ./
[ ! -e utils ] && cp -r  $KALDI_ROOT/egs/wsj/s5/utils ./ && cp $EESEN_ROOT/asr_egs/wsj/utils/* ./utils/ && cp $EESEN_ROOT/asr_egs/librispeech/utils/* ./utils/
[ ! -e local ] && ln -sf ../wsj/local ./
[ ! -e conf ] && ln -sf ../../conf ./
[ ! -e bin ] && ln -sf ../../bin ./

return 0

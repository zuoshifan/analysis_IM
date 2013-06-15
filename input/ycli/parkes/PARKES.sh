#! /bin/bash

export PIPELINE_DIR="/cita/h/home-2/ycli/analysis_IM/input/ycli/"
cd $PIPELINE_DIR
source $PIPELINE_DIR/PARAMS.sh

pipeline=parkes/parkes_pre.pipe

#===== Parkes Analysis ======
python $PIPELINE_DIR/manager.py $PIPELINE_DIR/$pipeline 

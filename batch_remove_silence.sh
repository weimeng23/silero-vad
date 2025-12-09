#!/bin/bash
# File              : batch_remove_silence.sh
# Author            : Meng Wei <wmeng94@gmail.com>
# Date              : 09.12.2025
# Last Modified Date: 09.12.2025
# Last Modified By  : Meng Wei <wmeng94@gmail.com>

input_dir=$1


python sf_remove_silence.py $input_dir -j 20 --overwrite -r

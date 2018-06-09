LEN_TRAIN='wc -l ../../../Data/sorted_sums_matched_struc_train_reversed.csv'
len_train=`$LEN_TRAIN | cut -f1 -d' '`
len_train_final=`expr $len_train - 1`

LEN_VAL='wc -l ../../../Data/sorted_sums_matched_struc_val.csv'
len_val=`$LEN_VAL | cut -f1 -d' '`
len_val_final=`expr $len_val - 1`

LEN_TEST='wc -l ../../../Data/sorted_sums_matched_struc_test.csv'
len_test=`$LEN_TEST | cut -f1 -d' '`
len_test_final=`expr $len_test - 1`

python training/training_mmnet.py ../../../Data/sorted_sums_matched_struc_train.csv ../../../Data/struc_data_reversed.svmlight ../../../Data/vocab.csv mmnet 3 $len_train_final $len_val_final $len_test_final --patience 1 --gpu --struc-aux-loss-wt 0 --conv-aux-loss-wt 0.0616 --embed-dropout-bool True --embed-dropout-p 0.433 --batch-size 16 --kernel-sizes 7,3,5 --num-filter-maps 391 --fc-dropout-p 0.409 --embed-file ../../../Data/processed_full.embed --struc-layer-size-list 419,29 --struc-activation relu --struc-dropout-list 0.3188 --bce-weights 0.5,1

python3 batch_test.py -c Wayside_EXP/c1.yml -e age20_base
python3 batch_test.py -c Wayside_EXP/c2.yml -e age20_base_2stage
python3 batch_test.py -c Wayside_EXP/c3.yml -e age20_base_NDT
python3 batch_test.py -c Wayside_EXP/c4.yml -e age20_base_NDT_2stage
mv ./tracker_exp_local/KITTI ./tracker_exp_local/0624_WAYSIDE_age20
python3 batch_test.py -c Wayside_EXP/c1_40.yml -e age40_base
python3 batch_test.py -c Wayside_EXP/c2_40.yml -e age40_base_2stage
python3 batch_test.py -c Wayside_EXP/c3_40.yml -e age40_base_NDT
python3 batch_test.py -c Wayside_EXP/c4_40.yml -e age40_base_NDT_2stage
mv ./tracker_exp_local/KITTI ./tracker_exp_local/0624_WAYSIDE_age40

# python3 batch_test.py -c KT_EXP/c1_40.yml -e age40_base
# python3 batch_test.py -c KT_EXP/c2_40.yml -e age40_base_2stage
# python3 batch_test.py -c KT_EXP/c3_40.yml -e age40_base_NDT
# python3 batch_test.py -c KT_EXP/c4_40.yml -e age40_base_NDT_2stage

# python3 batch_test.py -c KT_EXP/c1_60.yml -e age60_base
# python3 batch_test.py -c KT_EXP/c2_60.yml -e age60_base_2stage
# python3 batch_test.py -c KT_EXP/c3_60.yml -e age60_base_NDT
# python3 batch_test.py -c KT_EXP/c4_60.yml -e age60_base_NDT_2stage
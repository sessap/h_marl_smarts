# assumes knowing the exact env model
python3.8 run_learning.py scenarios/double_merge/lane_merge -f marl_benchmark/agents/dqn/baseline-lane-contro_truemodel.yaml --num_workers=7 --exp_name=lane_merge_known_model
# H_marl baseline
python3.8 run_learning.py scenarios/double_merge/lane_merge -f marl_benchmark/agents/dqn/baseline-lane-control.yaml --num_workers=7 --beta=1 - --exp_name=lane_merge_beta_1
# pred_mean baseline
python3.8 run_learning.py scenarios/double_merge/lane_merge -f marl_benchmark/agents/dqn/baseline-lane-control.yaml --num_workers=7 --beta=0 --exp_name=lane_merge_beta_0
# thompson sampling baseline
python3.8 run_learning.py scenarios/double_merge/lane_merge -f marl_benchmark/agents/dqn/baseline-lane-control.yaml --num_workers=7 --beta=1 --thompson --exp_name=lane_merge_thompson



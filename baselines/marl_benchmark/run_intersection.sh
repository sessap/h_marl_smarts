# assumes knowing the exact env model
python3.8 run_learning.py scenarios/intersections/4lane_onesv -f marl_benchmark/agents/dqn/baseline-lane-control_truemodel.yaml --num_workers=7 --exp_name=intersection_known_model
# H_marl baseline
python3.8 run_learning.py scenarios/intersections/4lane_onesv -f marl_benchmark/agents/dqn/baseline-lane-control.yaml --num_workers=7 --beta=1 --exp_name=intersection_beta_1
# pred_mean baseline
python3.8 run_learning.py scenarios/intersections/4lane_onesv -f marl_benchmark/agents/dqn/baseline-lane-control.yaml --num_workers=7 --beta=0 --exp_name=intersection_beta_0
# thompson sampling baseline
python3.8 run_learning.py scenarios/intersections/4lane_onesv -f marl_benchmark/agents/dqn/baseline-lane-control.yaml --num_workers=7 --beta=1 --thompson --exp_name=intersection_thompson

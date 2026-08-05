# DisentangledRNN
 Experimenting with the disentangled RNN from the NIPS article

* `disRNN_test.py` initial script to test the disRNN repo (https://github.com/google-deepmind/disentangled_rnns), reads `sbj1.mat`
	* `disRNN_ch_env.py` version to analyse data from Murphy et al. (reads `changing_environment_test.mat`, contains only part of the data)
	* `disRNN_test_trials.py` version to analyse the influence of the data size (runs on the cluster with `job-srun.sh`, generates `results.txt`)
* `analyze_no_of_trilals.py` reads `results.txt`an plots the results
* `disRNN_choice_conf.py` models choice and confidence by two categorical variables. However, the disRNN library supports only a single categorical output variable.

python main.py --config_file jiant/config/glue_lstm_large.conf --overrides "exp_name = glue-rnn-sst-qnli, pretrain_tasks = \"sst,qnli\", target_tasks = \"sst,qnli\""


python main.py --config_file jiant/config/glue_lstm_large.conf --overrides "exp_name = glue-rnn-sst-mnli, pretrain_tasks = \"sst,mnli\", target_tasks = \"sst,mnli\""


python main.py --config_file jiant/config/glue_lstm_large.conf --overrides "exp_name = glue-rnn-sst-qqp, pretrain_tasks = \"sst,qqp\", target_tasks = \"sst,qqp\""


python main.py --config_file jiant/config/glue_lstm_large.conf --overrides "exp_name = glue-rnn-qnli-mnli, pretrain_tasks = \"qnli,mnli\", target_tasks = \"qnli,mnli\""


python main.py --config_file jiant/config/glue_lstm_large.conf --overrides "exp_name = glue-rnn-qnli-qqp, pretrain_tasks = \"qnli,qqp\", target_tasks = \"qnli,qqp\""


python main.py --config_file jiant/config/glue_lstm_large.conf --overrides "exp_name = glue-rnn-mnli-qqp, pretrain_tasks = \"mnli,qqp\", target_tasks = \"mnli,qqp\""
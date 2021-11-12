from main import parser, arg_lists
from utils import str2bool

# add argument by group to global parser
###################################################

def process_config():
        # Argument
        data_arg = add_argument_group('Data')
        data_arg.add_argument('--data_dir', type=str, default='dataset/toefl/')
        data_arg.add_argument('--data_dir_cv', type=str, default='cv/')
        data_arg.add_argument('--num_cv_set', type=int, default=0)
        data_arg.add_argument('--gen_logs', type=str2bool, default=False)
        data_arg.add_argument('--log_dir', type=str, default='logs')
        data_arg.add_argument('--session_dir', type=str, default='session')
        data_arg.add_argument('--is_gen_cv', type=str2bool, default='False', help='Decide whether to generate new cv or reuse')  # disabled in this submission for reproduction
        data_arg.add_argument('--cur_fold', type=int, default=1, help='cur_fold for the cross validation')  # -1 indicates iterating whole folds
        data_arg.add_argument('--num_fold', type=int, default=5, help='the number of folds in Cross-Validation')  # e.g., 5 for asap
        data_arg.add_argument('--max_num_sents', type=int, default=-1, help='max length of document, will be defined by dataset')
        data_arg.add_argument('--max_len_sent', type=int, default=-1, help='max length of sentence, will be defined by dataset')
        data_arg.add_argument('--padding_place', type=str, default='post')
        data_arg.add_argument('--max_vocab_cnt', type=int, default=4000, help="if num of vocab is exceeded, then filter by freq")
        data_arg.add_argument('--keep_pronoun', type=str2bool, default=False)
        data_arg.add_argument('--remove_stopwords', type=str2bool, default=False)
        data_arg.add_argument('--essay_prompt_id_train', type=int, default=3)  # prompt id for essay corpus, from 1 to 8
        data_arg.add_argument('--essay_prompt_id_test', type=int, default=3)  # prompt id for essay corpus, from 1 to 8
        data_arg.add_argument('--pad_level', type=str, default='doc', help="padding level")  # "sent" or "doc, sentence level padding or document level
        data_arg.add_argument('--corpus_target', type=str, default="")  # asap, reada, gcdc, toefl  ## given by path if it is empty
        data_arg.add_argument('--gcdc_domain', type=str, default='Clinton')  # Clinton, Enron, Yahoo, Yelp

        data_arg.add_argument('--cv_attempts', type=int, default=1)  # the number of trial for Cross-Validation for reporting

        net_arg = add_argument_group('Network')
        net_arg.add_argument('--embed_size', type=int, default=100)  # for general pretariend, but emb for essay is pre-fixed in the code
        data_arg.add_argument('--tokenizer_type', type=str, default='xlnet-base-cased', help="types of tokenizer")  # "word", "bert-base-uncased", or "xlnet-base-cased"

        net_arg.add_argument('--path_pretrained_emb', type=str, default='xlnet-base-cased')  # path or "bert-base-uncased"
        net_arg.add_argument('--rnn_cell_type', type=str, default='gru', help='lstm, gru, qrnn')  # might be ignored as encoder_type
        net_arg.add_argument('--rnn_num_layer', type=int, default=1, help='# of layer used in rnn')
        net_arg.add_argument('--rnn_bidir', type=str2bool, default='false', help='lstm, gru, qrnn')  #
        net_arg.add_argument('--encoder_type', type=str, default='gru', help='lstm, gru, drnn, bert, reg_lstm, transf, xlnet')

        net_arg.add_argument('--rnn_cell_size', type=int, default=300)  # can be ignored as encoder (e.g., bert)
        net_arg.add_argument('--max_grad_norm', type=float, default=1)  # param for torch.nn.utils.clip_grad_norm
        net_arg.add_argument('--output_size', type=int, default=-1)  # if -1, the number of output class will be given in corpus class (GCDC: 3, ASAP: 1)
        net_arg.add_argument('--target_model', type=str, default="ilcr_kld", help='emnlp18, conll17, aaai18, ilcr_avg, ilcr_kld')

        train_arg = add_argument_group('Training')
        train_arg.add_argument('--op', type=str, default='adam')  # adam, sgd, asgd, rmsprop
        train_arg.add_argument('--eps', type=float, default=1e-8)
        train_arg.add_argument('--step_size', type=int, default=1)
        # train_arg.add_argument('--init_lr', type=float, default=0.0003)  # for xlnet
        train_arg.add_argument('--init_lr', type=float, default=0.003)
        train_arg.add_argument('--momentum', type=float, default=0.0)
        train_arg.add_argument('--lr_decay', type=float, default=0.0)  # 5e-4?
        train_arg.add_argument('--warmup_steps', type=int, default=0) # 100?
        train_arg.add_argument('--dropout', type=float, default=0.1)
        train_arg.add_argument('--improve_threshold', type=float, default=0.996)
        train_arg.add_argument('--patient_increase', type=float, default=4.0)
        train_arg.add_argument('--early_stop', type=str2bool, default=False)
        train_arg.add_argument('--max_epoch', type=int, default=25)
        train_arg.add_argument('--loss_type', type=str, default="CrossEntropyLoss")  # CrossEntropyLoss, MSELoss, nLLloss ...
        train_arg.add_argument('--eval_type', type=str, default="accuracy")  # accuracy, qwk

        train_arg.add_argument('--fp16_mode', type=str2bool, default=False)

        transf_arg = add_argument_group('Transfomer')
        transf_arg.add_argument('-d_model', type=int, default=128)
        transf_arg.add_argument('-d_inner_hid', type=int, default=128)
        transf_arg.add_argument('-d_k', type=int, default=64)
        transf_arg.add_argument('-d_v', type=int, default=64)
        transf_arg.add_argument('-n_head', type=int, default=8)
        transf_arg.add_argument('-transf_n_layers', type=int, default=6)
        transf_arg.add_argument('-embs_share_weight', action='store_true')
        transf_arg.add_argument('-proj_share_weight', action='store_true')

        misc_arg = add_argument_group('Misc')
        #misc_arg.add_argument('--save_model', type=str2bool, default=True)
        misc_arg.add_argument('--save_model', type=str2bool, default=False)
        misc_arg.add_argument('--print_step', type=int, default=15)
        misc_arg.add_argument('--ckpt_step', type=int, default=33)
        misc_arg.add_argument('--batch_size', type=int, default=32)
        misc_arg.add_argument('--use_apex', type=str2bool, default=False)
        # misc_arg.add_argument('--use_gpu', type=str2bool, default=False)
        misc_arg.add_argument('--use_gpu', type=str2bool, default=True)  # automatically assigned
        # misc_arg.add_argument('--use_parallel', type=str2bool, default=False)
        misc_arg.add_argument('--use_parallel', type=str2bool, default=True)
        misc_arg.add_argument('--empty_cache', type=str2bool, default=True)
        misc_arg.add_argument('--n_gpu', type=int, default=0)  # automatically assigned
        misc_arg.add_argument('--device', type=str, default="cuda")  # automatically assigned

        specific_arg = add_argument_group('Specific')
        specific_arg.add_argument('--skip_start', type=int, default=3, help='param used in AAAI18')  # 3 in asap dataset
        specific_arg.add_argument('--skip_jump', type=int, default=50, help='param used in AAAI18')  # 50 in asap dataset
        specific_arg.add_argument('--dim_tensor_feat', type=int, default=5, help='param used in AAAI18')  # 5 in asap datase

        specific_arg.add_argument('--drnn_layer', type=int, default=2, help='param used in DRNN')  # 
        
        specific_arg.add_argument('--wdrop', type=float, default=0.1, help='param used in reg_lstm')  # 
        specific_arg.add_argument('--dropoute', type=float, default=0.1, help='param used in reg_lstm')  # 
        specific_arg.add_argument('--beta_ema', type=float, default=0, help='param used in temporal averaging')  # 0.99?

        specific_arg.add_argument('--sem_dim_size', type=int, default=50)  # dim for semantic vec used in Stru Attention
        specific_arg.add_argument('--pooling_sent', type=str, default="max")  # avg, max
        specific_arg.add_argument('--pooling_doc', type=str, default="max")  # avg, max
        # specific_arg.add_argument("--local_rank", default=0, type=int)  # for apex
#
def add_argument_group(name):
        arg = parser.add_argument_group(name)
        arg_lists.append(arg)
        return arg

#
def get_config():
        parser.add_argument("--local_rank", default=-1, type=int)  # -1: non-distributed or 0: distributed
        parser.add_argument("--world_size", default=1, type=int)  # 
        config, _ = parser.parse_known_args()
        return config, _


# xlnet-asap: 0.0005
# emnlp18-asap: 0.01
# gru-asap: 0.001

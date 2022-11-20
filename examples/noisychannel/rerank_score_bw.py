import os
from contextlib import redirect_stdout
from fairseq import options
from fairseq_cli import generate
from examples.noisychannel import rerank_options, rerank_utils
def score_bw(args):
    pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, backwards_preprocessed_dir, lm_preprocessed_dir=rerank_utils.get_directories(args.data_dir_name, args.num_rescore, args.gen_subset,args.gen_model_name, args.shard_id, args.num_shards, args.sampling,args.prefix_len, args.target_prefix_frac, args.source_prefix_frac,)
    scorer1_src = args.target_lang#en
    scorer1_tgt = args.source_lang#de
    score1_file = pre_gen+"/bw_model_ex_score_translations.txt"
    print("STEP 4: score the translations for model 1")
    if not os.path.exists(score1_file):
        model_param1 = [ "--path", args.score_model1,#backward_en2de.pt 
                        "--source-lang", scorer1_src, "--target-lang", scorer1_tgt, ]
        gen_param=["--batch-size", str(128), "--score-reference", "--gen-subset", "train"]
        rerank_data1=backwards_preprocessed_dir
        gen_model1_param=[rerank_data1] + gen_param + model_param1
        gen_parser=options.get_generation_parser()
        input_args=options.parse_args_and_arch(gen_parser,gen_model1_param)
        with open(score1_file, "w") as f:
            with redirect_stdout(f):
                generate.main(input_args)
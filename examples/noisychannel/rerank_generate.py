import os
import subprocess
from contextlib import redirect_stdout
from fairseq import options
from fairseq_cli import generate, preprocess
from examples.noisychannel import rerank_utils
def gen_and_reprocess_nbest(args):
    pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, backwards_preprocessed_dir, lm_preprocessed_dir=rerank_utils.get_directories(args.data_dir_name, args.num_rescore, args.gen_subset,args.gen_model_name, args.shard_id, args.num_shards, args.sampling,args.prefix_len, args.target_prefix_frac, args.source_prefix_frac,)
    scorer1_src = args.target_lang
    scorer1_tgt = args.source_lang
    store_data = (
        os.path.join(os.path.dirname(__file__)) + "/rerank_data/" + args.data_dir_name
    )
    if not os.path.exists(store_data):
        os.makedirs(store_data)
    if not os.path.exists(pre_gen):
        os.makedirs(pre_gen)
    if not os.path.exists(left_to_right_preprocessed_dir):
        os.makedirs(left_to_right_preprocessed_dir)
    if not os.path.exists(right_to_left_preprocessed_dir):
        os.makedirs(right_to_left_preprocessed_dir)
    if not os.path.exists(lm_preprocessed_dir):
        os.makedirs(lm_preprocessed_dir)
    if not os.path.exists(backwards_preprocessed_dir):
        os.makedirs(backwards_preprocessed_dir)
    #score1_file=rerank_utils.rescore_file_name(pre_gen,args.prefix_len,args.model1_name,args.target_prefix_frac,args.source_prefix_frac,args.backwards1)
    using_nbest = args.nbest_list
    predictions_bpe_file = pre_gen + "/generate_output_bpe.txt"
    print("STEP 1: generate predictions using the p(T|S) model with bpe")
    if not os.path.exists(predictions_bpe_file):
        param1=[args.data,"--path",args.gen_model,"--shard-id",str(args.shard_id),"--num-shards",str(args.num_shards),
                "--nbest",str(args.num_rescore),"--batch-size",str(args.batch_size),"--beam",str(args.num_rescore),"--batch-size",
                str(args.num_rescore),"--gen-subset",args.gen_subset,"--source-lang",args.source_lang,"--target-lang",args.target_lang,]
        gen_parser=options.get_generation_parser()
        input_args=options.parse_args_and_arch(gen_parser, param1)
        with open(predictions_bpe_file, "w") as f:
            with redirect_stdout(f):
                generate.main(input_args)
    gen_output=rerank_utils.BitextOutputFromGen(predictions_bpe_file, args.post_process, using_nbest, args.prefix_len, args.target_prefix_frac)
    
    print("STEP 2: process the output of generate.py so we have clean text files with the translations")
    rescore_file = "/rescore_data"
    source_reproces=pre_gen +"/rescore_data.de"
    target_reproces=pre_gen +"/reference_file"
    hypo_reproces=pre_gen +"/rescore_data.en"
    if not (os.path.exists(source_reproces) or os.path.exists(target_reproces) or os.path.exists(hypo_reproces)):
        rerank_utils.write_reprocessed(gen_output.source, gen_output.hypo, gen_output.target,source_reproces, hypo_reproces, target_reproces, args.post_process)
    
    print("STEP 3: binarize the translations")
    bw_dict = args.data
    print("preprocess at %s"%(backwards_preprocessed_dir))
    bw_preprocess_param=[ "--source-lang", scorer1_src, "--target-lang", scorer1_tgt, "--trainpref", pre_gen + rescore_file, "--srcdict", bw_dict + "/dict." + scorer1_src + ".txt", "--tgtdict", bw_dict + "/dict." + scorer1_tgt + ".txt", "--destdir", backwards_preprocessed_dir, ]
    preprocess_parser=options.get_preprocessing_parser()
    input_args=preprocess_parser.parse_args(bw_preprocess_param)
    if not (os.path.exists(backwards_preprocessed_dir + "/dict." + scorer1_src + ".txt") or os.path.exists(backwards_preprocessed_dir + "/dict." + scorer1_src + ".txt")):
        preprocess.main(input_args)
    print("preprocess at %s"%(left_to_right_preprocessed_dir))
    preprocess_param = [ "--source-lang", scorer1_src, "--target-lang", scorer1_tgt, "--trainpref", pre_gen + rescore_file, "--srcdict", bw_dict + "/dict." + scorer1_src + ".txt", "--tgtdict", bw_dict + "/dict." + scorer1_tgt + ".txt", "--destdir", left_to_right_preprocessed_dir,]
    preprocess_parser=options.get_preprocessing_parser()
    input_args=preprocess_parser.parse_args(preprocess_param)
    if not (os.path.exists(left_to_right_preprocessed_dir + "/dict." + scorer1_src + ".txt") or os.path.exists(left_to_right_preprocessed_dir + "/dict." + scorer1_src + ".txt")):
        preprocess.main(input_args)
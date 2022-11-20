import os
from fairseq import options
from examples.noisychannel import rerank_options, rerank_utils
def score_lm(args):
    pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, backwards_preprocessed_dir, lm_preprocessed_dir=rerank_utils.get_directories(args.data_dir_name, args.num_rescore, args.gen_subset,args.gen_model_name, args.shard_id, args.num_shards, args.sampling,args.prefix_len, args.target_prefix_frac, args.source_prefix_frac,)
    predictions_bpe_file=pre_gen+"/generate_output_bpe.txt"
    nbest=False
    gen_output=rerank_utils.BitextOutputFromGen(predictions_bpe_file,args.post_process, nbest)
    lm_score_file = pre_gen + "/lm_score_translations_model_en_newscrawl.txt"
    print("STEP 4.5: language modeling for P(T)")
    if not os.path.exists(lm_score_file):
        bpe_status = "shared"
        rerank_utils.lm_scoring(lm_preprocessed_dir,bpe_status,gen_output,pre_gen,args.lm_dict,args.lm_name,args.language_model,args.lm_bpe_code,128,lm_score_file,args.target_lang,args.source_lang,args.prefix_len)
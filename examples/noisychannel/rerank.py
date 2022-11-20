import math
from multiprocessing import Pool
import numpy as np
from fairseq import options
from fairseq.data import dictionary
from fairseq.scoring import bleu
from examples.noisychannel import (
    rerank_generate,
    rerank_options,
    rerank_score_bw,
    rerank_score_lm,
    rerank_utils,
)
def load_score_files(args):
    pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, backwards_preprocessed_dir, lm_preprocessed_dir=rerank_utils.get_directories(args.data_dir_name, args.num_rescore, args.gen_subset,args.gen_model_name, args.shard_id, args.num_shards, args.sampling,args.prefix_len, args.target_prefix_frac, args.source_prefix_frac,)
    bw_score_file=pre_gen+"/bw_model_ex_score_translations.txt"
    lm_score_file=pre_gen+"/lm_score_translations_model_en_newscrawl.txt"
    predictions_bpe_file = pre_gen + "/generate_output_bpe.txt"
    using_nbest=True
    gen_output=rerank_utils.BitextOutputFromGen(predictions_bpe_file, args.post_process, using_nbest, args.prefix_len, args.target_prefix_frac)
    score1_file=rerank_utils.BitextOutput(bw_score_file, args.backwards1,args.right_to_left1,args.post_process,args.prefix_len,args.target_prefix_frac,args.source_prefix_frac)
    lm_res1=rerank_utils.LMOutput(lm_score_file, args.lm_dict, args.prefix_len, args.post_process, args.target_prefix_frac)
    
    return gen_output, score1_file, lm_res1
def score_target_hypo(args,weight1, weight2, weight3, lenpen, target_address, hypos_target,write_trigger,normalize):
    #weight2=0.929
    gen_output, bitext1, lm_res = load_score_files(args)
    dict = dictionary.Dictionary()
    scorer = scorer = bleu.Scorer(
        bleu.BleuConfig(
            pad=dict.pad(),
            eos=dict.eos(),
            unk=dict.unk(),
        )
    )
    ordered_hypos = {}
    ordered_targets = {}
    hypothesis_num=len(bitext1.rescore_source.keys())
    best_score= -math.inf
    j=1
    source_lst = []
    hypo_lst = []
    score_lst = []
    reference_lst = []
    for i in range(hypothesis_num):
        bitext2_score=gen_output.rescore_score[i]
        target_len = len(bitext1.rescore_hypo[i].split())
        lm_score=lm_res.score[i]
        src_len=bitext1.source_lengths[i]
        tgt_len=bitext1.target_lengths[i]
        bitext1_backwards = bitext1.backwards
        bitext2_backwards = gen_output.backwards
        score=rerank_utils.get_score(weight1, weight2, weight3, target_len, bitext1.rescore_score[i], bitext2_score, lm_score, lenpen, src_len, tgt_len, bitext1_backwards, bitext2_backwards, normalize)
        if score > best_score:
            best_score = score
            best_hypo = bitext1.rescore_hypo[i]
        if j == gen_output.num_hypos[i] or j == args.num_rescore:
            j=1
            hypo_lst.append(best_hypo)
            score_lst.append(best_score)
            source_lst.append(bitext1.rescore_source[i])
            reference_lst.append(bitext1.rescore_target[i])
            best_score = -math.inf
            best_hypo = ""
        else:
            j += 1
    gen_keys = list(sorted(gen_output.no_bpe_target.keys()))

    for key in range(len(gen_keys)):
        sys_tok = dict.encode_line(hypo_lst[key])
        ref_tok = dict.encode_line(gen_output.no_bpe_target[key])
        scorer.add(ref_tok, sys_tok)
        if write_trigger:
            ordered_hypos[key] = hypo_lst[key]
            ordered_targets[key] = gen_output.no_bpe_target[key]
    if write_trigger:
        with open(target_address, "w") as t:
            with open(hypos_target, "w") as h:
                for key in range(len(ordered_hypos)):
                    t.write(ordered_targets[key])
                    h.write(ordered_hypos[key])
    res = scorer.result_string(4)
    print(res)
    score = rerank_utils.parse_bleu_scoring(res)
    return score
    
def match_target_hypo(args, write_targets, write_hypos):
    rerank_score=score_target_hypo(args,args.weight1[0], args.weight2[0], args.weight3[0], args.lenpen[0], write_targets, write_hypos, True, args.normalize)
    return args.lenpen, args.weight1, args.weight2, args.weight3, rerank_score
def rerank(args):
    pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, backwards_preprocessed_dir, lm_preprocessed_dir=rerank_utils.get_directories(args.data_dir_name, args.num_rescore, args.gen_subset,args.gen_model_name, args.shard_id, args.num_shards, args.sampling,args.prefix_len, args.target_prefix_frac, args.source_prefix_frac,)
    rerank_generate.gen_and_reprocess_nbest(args)
    rerank_score_bw.score_bw(args)
    rerank_score_lm.score_lm(args)
    write_targets= pre_gen + "/matched_targets"
    write_hypos= pre_gen + "/matched_hypos"
    lenpen, weight1, weight2, weight3, score=match_target_hypo(args, write_targets, write_hypos)
    
    return lenpen, weight1, weight2, weight3, score
def cli_main():
    parser=rerank_options.get_reranking_parser()
    args=options.parse_args_and_arch(parser)
    rerank(args)
if __name__ == "__main__":
	cli_main()

    
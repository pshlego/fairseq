import math
import os
import re
import subprocess
from contextlib import redirect_stdout
from fairseq import options
from fairseq_cli import eval_lm, preprocess
def get_score(weight1, weight2, weight3, target_len, channel_model_score,direct_model_score, lm_score, lenpen, src_len, tgt_len, bitext1_backwards, bitext2_backwards, normalize):
    score=weight1*channel_model_score+weight2*direct_model_score+weight3*lm_score
    score /= (target_len) ** float(lenpen)
    return score
def write_reprocessed(source, hypo, target,source_reproces, hypo_reproces, target_reproces, bpe_symbol):
    if not os.path.exists(source_reproces):
        with open(source_reproces, 'w') as file1:
            with redirect_stdout(file1):
                for i in sorted(source):
                    for _ in range(len(hypo[0])):
                        file1.writelines(source[i])
    if not os.path.exists(target_reproces):
        with open(target_reproces, 'w') as file2:
            with redirect_stdout(file2):
                for i in sorted(target):
                    for _ in range(len(hypo[0])):
                        file2.writelines(target[i])
    if not os.path.exists(hypo_reproces):
        with open(hypo_reproces, 'w') as file3:
            with redirect_stdout(file3):
                for i in sorted(hypo):
                    for j in range(len(hypo[i])):
                        file3.writelines(hypo[i][j])
def parse_bleu_scoring(line):
    p = re.compile(r"(BLEU4 = )\d+[.]\d+")
    res = re.search(p, line)
    return float(res.group()[8:])
def reprocess(fle):
    with open(fle, "r") as f:
        txt = f.read()
    p = re.compile(r"[STHP][-]\d+\s*")# sentence의 종류를 알기 위한 filter
    hp = re.compile(r"(\s*[-]?\d+[.]?\d+\s*)|(\s*(-inf)\s*)")#hypothesis의 점수임을 확인하기 위한 filter
    source_dict = {}
    hypothesis_dict = {}
    score_dict = {}
    target_dict = {}
    pos_score_dict = {}
    lines = txt.split("\n")

    for line in lines:
        line += "\n"
        prefix = re.search(p, line)
        if prefix is not None:
            _, j = prefix.span()
            id_num = prefix.group()[2:]
            id_num = int(id_num)
            line_type = prefix.group()[0]
            if line_type == "S":
                source_dict[id_num] = line[j:]
            elif line_type == "T":
                target_dict[id_num] = line[j:]
            elif line_type == "H":
                h_txt = line[j:]
                hypo = re.search(hp, h_txt)
                _, i = hypo.span()
                score = hypo.group()
                if id_num in hypothesis_dict:
                    hypothesis_dict[id_num].append(h_txt[i:])
                    score_dict[id_num].append(float(score))
                else:
                    hypothesis_dict[id_num] = [h_txt[i:]]
                    score_dict[id_num] = [float(score)]
            elif line_type == "P":
                pos_scores = (line[j:]).split()
                pos_scores = [float(x) for x in pos_scores]
                if id_num in pos_score_dict:
                    pos_score_dict[id_num].append(pos_scores)
                else:
                    pos_score_dict[id_num] = [pos_scores]

    return source_dict, hypothesis_dict, score_dict, target_dict, pos_score_dict
def get_directories(data_dir_name, num_rescore, gen_subset,gen_model_name, shard_id, num_shards, sampling,prefix_len, target_prefix_frac, source_prefix_frac):
    train_id = '/nbest_'+str(num_rescore)+'_subset_'+gen_subset+"_fw_name_" + gen_model_name+"_shard_" +str(shard_id)+"_of_"+str(num_shards)
    pre_gen = os.path.dirname(os.path.realpath(__file__))+"/rerank_data/"+data_dir_name+train_id
    left_to_right_preprocessed_dir=pre_gen+'/left_to_right_preprocessed'
    right_to_left_preprocessed_dir=pre_gen+'/right_to_left_preprocessed'
    backwards_preprocessed_dir=pre_gen+'/backwards'
    lm_preprocessed_dir=pre_gen+'/lm_preprocessed'
    return pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, backwards_preprocessed_dir, lm_preprocessed_dir
def remove_bpe(line, bpe_symbol):
    line = line.replace("\n", "")
    line = (line + " ").replace(bpe_symbol, "").rstrip()
    return line + ("\n")
def remove_bpe_dict(pred_dict, bpe_symbol):
    new_dict = {}
    for i in pred_dict:
        if type(pred_dict[i]) == list:
            new_list = [remove_bpe(elem, bpe_symbol) for elem in pred_dict[i]]
            new_dict[i] = new_list
        else:
            new_dict[i] = remove_bpe(pred_dict[i], bpe_symbol)
    return new_dict
def get_score_from_pos(pos_score_dict, prefix_len, hypo_dict, bpe_symbol, hypo_frac, backwards):
    score_dict = {}
    num_bpe_tokens_dict = {}
    for key in pos_score_dict:
        score_dict[key] = []
        num_bpe_tokens_dict[key] = []
        for i in range(len(pos_score_dict[key])):
            score_dict[key].append(sum(pos_score_dict[key][i]))
            num_bpe_tokens_dict[key].append(len(pos_score_dict[key][i]))
    return score_dict, num_bpe_tokens_dict
class BitextOutputFromGen(object):
    def __init__(
        self,
        predictions_bpe_file,
        bpe_symbol=None,
        nbest=False,
        prefix_len=None,
        target_prefix_frac=None,
    ):
        pred_source, pred_hypo, pred_score, pred_target, pred_pos_score = reprocess(predictions_bpe_file)
        pred_score, num_bpe_tokens = get_score_from_pos(
            pred_pos_score, prefix_len, pred_hypo, bpe_symbol, target_prefix_frac, False
        )#POS를 다 더함으로써 leng penalty를 없앤다.
        self.source = pred_source
        self.target = pred_target
        self.score = pred_score
        self.pos_score = pred_pos_score
        self.hypo = pred_hypo
        self.no_bpe_source = remove_bpe_dict(pred_source.copy(), bpe_symbol)
        self.no_bpe_hypo = remove_bpe_dict(pred_hypo.copy(), bpe_symbol)
        self.no_bpe_target = remove_bpe_dict(pred_target.copy(), bpe_symbol)
        self.target_lengths = {}
        self.source_lengths = {}
        self.rescore_source = {}
        self.rescore_target = {}
        self.rescore_pos_score = {}
        self.rescore_hypo = {}
        self.rescore_score = {}
        self.num_hypos = {}
        self.backwards = False
        self.right_to_left = False
        index = 0
        for i in sorted(self.source.keys()):
            for j in range(len(self.hypo[i])):
                self.target_lengths[index] = len(self.hypo[i][j].split())
                self.source_lengths[index] = len(self.source[i].split())
                self.num_hypos[index] = len(self.hypo[i])
                self.rescore_source[index] = self.no_bpe_source[i]
                self.rescore_target[index] = self.no_bpe_target[i]
                self.rescore_hypo[index] = self.no_bpe_hypo[i][j]
                self.rescore_score[index] = float(self.score[i][j])
                self.rescore_pos_score[index] = self.pos_score[i][j]
                index += 1
class BitextOutput:
    def __init__(self,score1_file, backwards1,right_to_left1,bpe_symbol,prefix_len,target_prefix_frac,source_prefix_frac):
        hypo,source,score, target, pos_score=reprocess(score1_file)
        self.hypo_fracs = source_prefix_frac
        score, num_bpe_tokens = get_score_from_pos(pos_score, prefix_len, hypo, bpe_symbol, self.hypo_fracs, backwards1)
        source_lengths = {}
        target_lengths = {}
        for i in source:
            len_src = len(source[i][0].split())
            #<eos>를 없애고 저장한다.               
            source_lengths[i] = num_bpe_tokens[i][0] - 1
            target_lengths[i] = len(hypo[i].split())
            source[i] = remove_bpe(source[i][0], bpe_symbol)
            target[i] = remove_bpe(target[i], bpe_symbol)
            hypo[i] = remove_bpe(hypo[i], bpe_symbol)
            score[i] = float(score[i][0])
            pos_score[i] = pos_score[i][0]
            self.rescore_source = source
        self.rescore_hypo = hypo
        self.rescore_score = score
        self.rescore_target = target
        self.rescore_pos_score = pos_score
        self.backwards = backwards1
        self.right_to_left = right_to_left1
        self.target_lengths = target_lengths
        self.source_lengths = source_lengths
def parse_lm(lm_score_file, prefix_len, bpe_symbol, target_prefix_frac):
    with open(lm_score_file, "r") as f:
        text = f.readlines()
        text = text[4:]
        cleaned_text = text[:-2]
        sentences = {}
        sen_scores = {}
        sen_pos_scores = {}
        no_bpe_sentences = {}
        num_bpe_tokens_dict = {}
        for _i, line in enumerate(cleaned_text):
            tokens = line.split()
            if tokens[0].isdigit():#문자열이 숫자열인지 확인
                line_id = int(tokens[0])
                scores = [float(x[1:-1]) for x in tokens[2::2]]#2씩 건너뛰어서 점수 획득
                sentences[line_id] = " ".join(tokens[1::2][:-1]) + "\n"
                bpe_sen = " ".join(tokens[1::2][:-1]) + "\n"
                no_bpe_sen = remove_bpe(bpe_sen, bpe_symbol)
                no_bpe_sentences[line_id] = no_bpe_sen
                sen_scores[line_id] = sum(scores)
                num_bpe_tokens_dict[line_id] = len(scores)
                sen_pos_scores[line_id] = scores
    return sentences, sen_scores, sen_pos_scores, no_bpe_sentences, num_bpe_tokens_dict
class LMOutput:
    def __init__(self,lm_score_file,lm_dict, prefix_len, bpe_symbol, target_prefix_frac):
        lm_sentences,lm_sen_scores,lm_sen_pos_scores,lm_no_bpe_sentences,lm_bpe_tokens=parse_lm(lm_score_file, prefix_len, bpe_symbol, target_prefix_frac)
        self.sentences = lm_sentences
        self.score = lm_sen_scores
        self.pos_score = lm_sen_pos_scores
        self.lm_dict = lm_dict
        self.no_bpe_sentences = lm_no_bpe_sentences
        self.bpe_tokens = lm_bpe_tokens
def lm_scoring(preprocess_directory,bpe_status,gen_output,pre_gen,cur_lm_dict,lm_name, cur_language_model,lm_bpe_code,batch_size,lm_score_file, target_lang,source_lang,prefix_len):
    preprocess_lm_param = [
            "--only-source",
            "--trainpref",
            pre_gen + "/rescore_data." + target_lang,#target sentences
            "--srcdict",
            cur_lm_dict,
            "--destdir",
            preprocess_directory,
        ]
    preprocess_parser = options.get_preprocessing_parser()
    input_args = preprocess_parser.parse_args(preprocess_lm_param)
    preprocess.main(input_args)
    eval_lm_param = [preprocess_directory, "--path", cur_language_model, "--output-word-probs", "--batch-size", str(batch_size), "--sample-break-mode", "eos", "--gen-subset", "train", ]
    eval_lm_parser=options.get_eval_lm_parser()
    input_args=options.parse_args_and_arch(eval_lm_parser, eval_lm_param)
    with open(lm_score_file, "w")as f:
        with redirect_stdout(f):
            eval_lm.main(input_args)
import cPickle as pickle
import logging
import gensim
from tqdm import tqdm
import os,sys

min_cnt = 20
NNsize = 200

#DATA_FOLDER = '../askubuntu/'

def buildWord2Vec(DATA_FOLDER): 
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # tokens = {'id':'token list'}
    question_body_tokens = pickle.load(open(DATA_FOLDER + 'tokenized_question_body.pkl', 'rb'))
    question_title_tokens = pickle.load(open(DATA_FOLDER + 'tokenized_question_title.pkl', 'rb'))
    answer_body_tokens = pickle.load(open(DATA_FOLDER + 'tokenized_answer_body.pkl', 'rb'))

    sentences = []
    for q in tqdm(question_title_tokens):
        sentences.append(question_title_tokens[q])
    for q in tqdm(question_body_tokens):
        sentences.append(question_body_tokens[q])
    for a in tqdm(answer_body_tokens):
        sentences.append(answer_body_tokens[a])
    print len(sentences)
    model = gensim.models.Word2Vec(sentences, min_count=min_cnt, size=NNsize, workers=4, iter=5)
    fname = DATA_FOLDER + 'word2vec_' + 'min_cnt_' + str(min_cnt) + 'NNsize_' + str(NNsize) 
    model.save(fname)

    
def buildWord2Vec_multi(DATA_FOLDER, DATA_FOLDER_2, SAVE_FOLDER): 
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # tokens = {'id':'token list'}
    question_body_tokens = pickle.load(open(DATA_FOLDER + 'tokenized_question_body.pkl', 'rb'))
    question_title_tokens = pickle.load(open(DATA_FOLDER + 'tokenized_question_title.pkl', 'rb'))
    answer_body_tokens = pickle.load(open(DATA_FOLDER + 'tokenized_answer_body.pkl', 'rb'))
    
    question_body_tokens_2 = pickle.load(open(DATA_FOLDER_2 + 'tokenized_question_body.pkl', 'rb'))
    question_title_tokens_2 = pickle.load(open(DATA_FOLDER_2 + 'tokenized_question_title.pkl', 'rb'))
    answer_body_tokens_2 = pickle.load(open(DATA_FOLDER_2 + 'tokenized_answer_body.pkl', 'rb'))

    
    sentences = []
    for q in tqdm(question_title_tokens):
        sentences.append(question_title_tokens[q])
    for q in tqdm(question_body_tokens):
        sentences.append(question_body_tokens[q])
    for a in tqdm(answer_body_tokens):
        sentences.append(answer_body_tokens[a])
        
        
    for q in tqdm(question_title_tokens_2):
        sentences.append(question_title_tokens_2[q])
    for q in tqdm(question_body_tokens_2):
        sentences.append(question_body_tokens_2[q])
    for a in tqdm(answer_body_tokens_2):
        sentences.append(answer_body_tokens_2[a])
        
    print len(sentences)
    model = gensim.models.Word2Vec(sentences, min_count=min_cnt, size=NNsize, workers=4, iter=5)
    fname = SAVE_FOLDER + 'word2vec_' + 'min_cnt_' + str(min_cnt) + 'NNsize_' + str(NNsize) 
    model.save(fname)
    
if __name__ == '__main__':
    if len(sys.argv) == 2:
        DATA_FOLDER = sys.argv[1]
        if not os.path.isdir(DATA_FOLDER):
            print 'not valid data folder...'
        else: 
            buildWord2Vec(DATA_FOLDER)
    elif len(sys.argv) == 4:
        DATA_FOLDER = sys.argv[1]
        DATA_FOLDER_2 = sys.argv[2]
        SAVE_FOLDER = sys.argv[3]
        if (not os.path.isdir(DATA_FOLDER)) or (not os.path.isdir(DATA_FOLDER_2) or (not os.path.isdir(SAVE_FOLDER))):
            print 'not valid data folder...'
        else: 
            buildWord2Vec_multi(DATA_FOLDER, DATA_FOLDER_2, SAVE_FOLDER)
    else:
        print 'input data folder...'
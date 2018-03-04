import cPickle as pickle
import nltk
import re
from tqdm import tqdm 
import os,sys


def buildTokens(DATA_FOLDER):
    QuestionList = pickle.load(open(DATA_FOLDER + 'Questions.pkl', 'rb'))
    AnswerList = pickle.load(open(DATA_FOLDER + 'Answers.pkl', 'rb'))

    tokenized_question_body = {}
    tokenized_question_title = {}
    tokenized_answer_body = {}

    # stop_words = set(nltk.corpus.stopwords.words("english"))
    # To do: filter <label>

    TAG_RE = re.compile(r'<[^>]+>')

    def remove_tags(text):
        return TAG_RE.sub('', text).lower()


    for q in tqdm(QuestionList.keys()):
        body_sentence = remove_tags(QuestionList[q].body)
        title_sentence = remove_tags(QuestionList[q].title)
        body_tokens = nltk.word_tokenize(body_sentence)
        title_tokens = nltk.word_tokenize(title_sentence)
        tokenized_question_body[q] = body_tokens
        tokenized_question_title[q] = title_tokens

    for a in tqdm(AnswerList.keys()):
        sentence = remove_tags(AnswerList[a].body)
        tokens = nltk.word_tokenize(sentence)
        tokenized_answer_body[a] = tokens 

    pickle.dump(tokenized_question_body, open(DATA_FOLDER + 'tokenized_question_body.pkl', 'wb'))
    pickle.dump(tokenized_question_title, open(DATA_FOLDER + 'tokenized_question_title.pkl', 'wb'))
    pickle.dump(tokenized_answer_body, open(DATA_FOLDER + 'tokenized_answer_body.pkl', 'wb'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_FOLDER = sys.argv[1]
        if not os.path.isdir(DATA_FOLDER):
            print 'not valid data folder...'
        else: 
            buildTokens(DATA_FOLDER)
    else:
        print 'input data folder...'
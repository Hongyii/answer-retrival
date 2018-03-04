import xml.etree.ElementTree as ET
import cPickle as pickle
from ..DataFormat.Question import Question
from ..DataFormat.Answer import Answer
from tqdm import tqdm
import os, sys


def buildQAlist(DATA_FOLDER):
    # read posts.xml file 
    tree = ET.parse(DATA_FOLDER +'Posts.xml')
    root = tree.getroot()
    pickle.dump(root, open(DATA_FOLDER + 'Posts.pkl', 'wb'))
    print 'Process xml finished...'
    os.remove(DATA_FOLDER +'Posts.xml')

    # build question and answer datastucture
    QUESTION_ATTRIB = ['AcceptedAnswerId', 'Body', 'Id', 'Tags', 'Title']
    ANSWER_ATTRIB = ['Body', 'Id', 'ParentId']
    QuestionList = {}
    AnswerList = {}
    #root = pickle.load(open(DATA_FOLDER +'Posts.pkl', 'rb'))

    for i in tqdm(range(0,len(root))):
        exp = root[i]
        data = []
        postType = exp.attrib['PostTypeId']
        if postType == '1':
            try:
                for e in QUESTION_ATTRIB :
                    data.append(exp.attrib[e])
                QuestionList[exp.attrib['Id']] = Question(data)
            except:
                continue
        if postType == '2':
            for e in ANSWER_ATTRIB :
                data.append(exp.attrib[e])
            AnswerList[exp.attrib['Id']] = Answer(data)

    pickle.dump(QuestionList, open(DATA_FOLDER + 'Questions.pkl', 'wb'))
    pickle.dump(AnswerList, open(DATA_FOLDER + 'Answers.pkl', 'wb'))


    #QuestionList = pickle.load(open(DATA_FOLDER + 'Questions.pkl', 'rb'))
    #AnswerList = pickle.load(open(DATA_FOLDER + 'Answers.pkl', 'rb'))


    # build QA pair and candidate answer list
    pair = {}
    for e in QuestionList:
        if QuestionList[e].answerId in AnswerList:
            pair[e] = QuestionList[e].answerId
    pickle.dump(pair, open(DATA_FOLDER + 'Pairs.pkl', 'wb'))

    QAList = {}
    counter = 0

    for q in pair:
        QAList[q] = []

    for a in AnswerList:
        parent = AnswerList[a].parentId
        if parent in QAList:
            QAList[parent].append(a)
        print 'answerlist', counter
        counter += 1

    pickle.dump(QAList, open(DATA_FOLDER + 'QAList.pkl', 'wb'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_FOLDER = sys.argv[1]
        if not os.path.isdir(DATA_FOLDER):
            print 'not valid data folder...'
        else: 
            buildQAlist(DATA_FOLDER)
    else:
        print 'input data folder...'
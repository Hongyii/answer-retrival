import xml_dump
import tokenize_dump
import word2vec

def pipeline(DATA_FOLDER):
    # extract QA data structure
    xml_dump.buildQAlist(DATA_FOLDER)
    # tokenize
    tokenize_dump.buildTokens(DATA_FOLDER)
    # word2vec model
    word2vec.buildWord2Vec(DATA_FOLDER)
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_FOLDER = sys.argv[1]
        if not os.path.isdir(DATA_FOLDER):
            print 'not valid data folder...'
        else: 
            buildTokens(DATA_FOLDER)
    else:
        print 'input data folder...'
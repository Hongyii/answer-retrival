import os
import cPickle as pickle
import sys

TB_FOLDER = '../multitask/tb/'
reg_list = [1e-4,1e-5, 1e-6, 1e-7]
lstm_size = [100, 200, 300]
batch_size = {100:1000, 200:500, 300:350}

def createTask():
    Task = []
    for reg in reg_list:
        for lstm in lstm_size:
            directory = TB_FOLDER + str(reg) +  '_' + str(lstm)
            task = {'reg':reg, 'lstm':lstm, 'batch': batch_size[lstm], 'dir': directory}
            #if not os.path.exists(directory):
             #   os.makedirs(directory)
            print directory
            Task.append(task)

    pickle.dump(Task, open('../multitask/task.pkl', 'wb')) 

def createSubTask():
    Task = pickle.load(open('../multitask/task.pkl', 'rb'))
    #print len(Task)
    for i in range(0, len(Task), 3):
        #print i
        subtask = [Task[i], Task[i+1], Task[i+2]]
        pickle.dump(subtask, open('../multitask/task'+str(i/3)+'.pkl', 'wb')) 
        
        
if __name__ == '__main__':
    #createTask()
    #createSubTask()
    task1_folder = sys.argv[1]
    task2_folder = sys.argv[2]
    task_file = sys.argv[3]
    tasks = pickle.load(open(task_file, 'rb'))
    for task in tasks:
        if not os.path.exists(task['dir']):
            os.makedirs(task['dir'])
        else:
            continue
        split = ' '
        cmd = 'python lstm_multitask.py ' + task1_folder + split +  task2_folder + split + str(task['dir']) + split + str(task['reg']) + split + str(task['lstm']) + split + str(task['batch'])
        print cmd
        os.system(cmd)

    
    
    
    

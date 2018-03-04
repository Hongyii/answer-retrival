import os
import cPickle as pickle
import sys

reg_list = [1e-4,1e-5, 1e-6, 1e-7]
lstm_size = [100, 200, 300]
batch_size = {100:1000, 200:500, 300:350}

def createTask(task_folder):
    Task = []
    for reg in reg_list:
        for lstm in lstm_size:
            directory = task_folder + 'tb/'+ str(reg) +  '_' + str(lstm)
            task = {'reg':reg, 'lstm':lstm, 'batch': batch_size[lstm], 'dir': directory}
            print directory
            Task.append(task)
    pickle.dump(Task, open(task_folder + 'task.pkl', 'wb')) 

def createSubTask(task_folder):
    Task = pickle.load(open(task_folder + 'task.pkl', 'rb'))
    for i in range(0, len(Task), 3):
        subtask = [Task[i], Task[i+1], Task[i+2]]
        pickle.dump(subtask, open(task_folder + 'task' + str(i/3)+'.pkl', 'wb')) 
        
if __name__ == '__main__':
    task_folder = sys.argv[1]
    #createTask(task_folder)
    #createSubTask(task_folder)
    task_file = sys.argv[2]
    tasks = pickle.load(open(task_file, 'rb'))
   
    for task in tasks:
        if not os.path.exists(task['dir']):
            os.makedirs(task['dir'])
        else:
            continue
        split = ' '
        cmd = 'python lstm_multilayer.py ' + task_folder + split + str(task['dir']) + split + str(task['reg']) + split + str(task['lstm']) + split + str(task['batch'])
        print cmd
        os.system(cmd)

    
    
    
    

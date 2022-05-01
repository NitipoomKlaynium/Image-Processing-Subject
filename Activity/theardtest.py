import threading 

# global variable x 
x = 0

def increment(): 
    """ 
    function to increment global variable x 
    """
    global x 
    x += 1

def thread_task(): 
    """ 
    task for thread 
    calls increment function 100000 times. 
    """
    global x
    while x < 1000000: 
        increment() 
    print("!!!!")

def main_task(): 
    global x 
    # setting global variable x as 0 
    x = 0

    # creating threads 
    t1 = threading.Thread(target=thread_task) 
    t2 = threading.Thread(target=thread_task) 

    # start threads 
    t1.start() 
    t2.start() 

    # wait until threads finish their job 
    print("123")
    t1.join() 
    print(x)
    t2.join() 
    print("456")

if __name__ == "__main__": 
        main_task() 
        print(x)
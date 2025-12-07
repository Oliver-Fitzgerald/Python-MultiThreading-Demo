import marimo

__generated_with = "0.18.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from rich import print
    from rich.console import Console
    from PIL import Image
    import threading
    import time
    import random
    import matplotlib.pyplot as plt
    from collections import defaultdict
    return Console, Image, defaultdict, mo, plt, print, random, threading, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **A Guide to Multi-threading in Python**

    This notebook explains how multi-threading can be utilized in python through practical examples that highligth common use cases and pit falls.

    ## **1. Threads Explained**

    threads (sometimes refered to as ligthweigth processes) are a sequence of instructions that can be executed independently of other code. Independantly in this context means with it's own TCB (Thread Control Block) and stack however they will share certain information with threads created from a common procces including code instructions, data and OS resources such as open files.


    ### TCB - Thread Control Block

    The thread control block is a data structure in operating systems which contains all of the nessecary information for tracking and and controlling the execution of a thread. It contains identifiers such as references to the threads parent PCB, TCB and the unique thread identifier and data specific to the thread such as thread state, stack pointer, priority, etc.

    multi-threading is therfore the practice of running multiple thrads within a proccess

    ## **2. Creating threads**

    In our first demo we write a simple script in which we will ustilize multithreading to allow a user to use an add function to sum up two different lists of numbers. Making progress on both lists at the same time through the process of interleaving that multi-threading facilitates.
    Threads in python can be created via the threading package with threading.
    Thread with the following parameters

    | Parameter |      Type       |Description|
    |-----------|-----------------|-----------|
    |   target  | Callback Object | Defines a target method for the thread to execute if None the Thread.run method will run wich can be overriden in a Thread child class |
    |    name   |      String     | The name of the thread defaults to Thread-N where N is the thread number (Usually for debugging)|
    |    args   |       tuple     | A tuple of positional arguments passed to target |
    |   kwargs  |       dict      | A dictionary of arguments passed to target
    |   daemon  |       Bool      | Determines wether a thread is a deamon thread or not (Will be explained in a later section) |

    **Interleaving** is the process of taking two or more  independant sequences of statements and executing their steps in a mixed order, where the final execution order is detemined by some scheduling alorithm or criteria

    Example:
    - Thread A: A1, A2, A3
    - Thread B: B1, B2, B3

    Possible Resulting Execution Orders
    [A1,A2,A3,B1,B2,B3]
    [B1,B2,B3,A1,A2,A3]
    [A1,B1,A2,A3,B2,B3]
    etc ...
    """)
    return


@app.cell
def _(print, threading):
    # Defining the programs varriables and functions
    A = [2,2,2,2,2]
    B = [4,4,4,4,4]
    results = {}

    def add(list_of_numbers, name) -> int:
        """Sums a list of numbers and returns the result"""
        results[name] = sum(list_of_numbers) 


    # Creating threads
    threadA = threading.Thread(target=add, name="Thread-A", args=(A,"Thread-A"))
    threadB = threading.Thread(target=add, name="Thread-B", args=(B,"Thread-B"))

    # We then use the start method to trigger the execution of the target method for each thread
    threadA.start()
    threadB.start()

    # We then call the join() method on each thread which forces the main thread to wait for the thread that join was called on to complete before procedding. This ensures that the add method has completed before we try and access it's result
    threadA.join()
    threadB.join()

    #Printing results
    print("Thread A final result: ", results["Thread-A"])
    print("Thread B final result: ", results["Thread-B"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating threads (Continued.)

    The following demo shows that both threads are indeed being interleved graphically through the use of progress bars.
    We start by creating a task class that will represent a unit of work for a thread to do and have each thread update a loading bar with the progression made on the given task.
    **NOTE:** That the progress bars are both being update simultaniously

    *Update the FINISH varriable to lower or higher if tasks are taking to long or to quick.*
    """)
    return


@app.cell
def _(mo, threading, time):
    FINISH = 1000000

    class Task:

        def __init__(self, start: int, finish: int, threadName: str):
            self.start = start
            self.finish = finish
            self.threadName = threadName
            self.progress = 0  # Shared counter

        def run(self):
            """
            counts the number of primes in a given range.
            returns -1 if range is invalid
            """
            if self.start >= self.finish:
                raise ValueError("start must preceed finish")

            itterations = 0
            count = 0 # the number of primes found

            for number in range(int(self.finish - self.start)):

                # progress tracker
                if itterations % ((self.finish - self.start) / 10) == 0:
                    self.progress = ((itterations / (self.finish - self.start)) * 100)

                # count primes
                if self._is_prime(number):
                    count+=1
                itterations += 1

            self.progress = 100


        def _is_prime(self,n):
            """
            Determines wether or not a passed number (n) is greater than 1 and has 2 factors or not.
            """
            if n < 2:
                return False
            else:
                for i in range(2, int(n ** 0.5) + 1):
                    if n % i == 0:
                        return False
            return True



    # Create two tasks
    task1 = Task(start=0, finish=FINISH, threadName="Thread-A")
    task2 = Task(start=0, finish=FINISH, threadName="Thread-B")

    # Start both threads
    _thread1 = threading.Thread(name="Thread-A", target=task1.run)
    _thread2 = threading.Thread(name="Thread-B", target=task2.run)

    _thread1.start()
    _thread2.start()

    # Create both progress bars with context managers
    with mo.status.progress_bar(total=100, title="Thread-1", subtitle="Counting primes ...") as bar1, \
         mo.status.progress_bar(total=100, title="Thread-2", subtitle="Counting primes ...") as bar2:

        last_progress1 = 0
        last_progress2 = 0

        while _thread1.is_alive() or _thread2.is_alive():
            time.sleep(0.05)
            # Update thread 1's progress bar
            if task1.progress > last_progress1:
                bar1.update(task1.progress - last_progress1)
                last_progress1 = task1.progress

            # Update thread 2's progress bar
            if task2.progress > last_progress2:
                bar2.update(task2.progress - last_progress2)
                last_progress2 = task2.progress


    _thread1.join()
    _thread2.join()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating threads (Continued.)

    **The daemon paramter**
    When creating a thread in python you can define wether or not a thread is a daemon thread or not. Daemon threads function to support the primary thread in the background and so therfore when the main threaed terminates so does the daemon thread. Non-daemon threads on the other hand is a thread tasked with some application critical work and as such will keep the main thread alive even after it has completed.

    Note that threads in python are non-daemon threads by default
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Race Conditions

    ### **The Problem**

    A race condition occurs when two or more threads that have been created attempt to access shared memory at the same time. As thread swapping algorithms can swap between threads at any given time, you cannot know the order in which the threads will access the shared data. Therfore the resulting value of the shared data will be dependant on the thread scheduling algorithm.

    The following code block demonstrates how a race-condition could occur. Note that the desired output is the number of ittereations 10000 and that the actual output is different each time you run the cell.
    """)
    return


@app.cell
def _(print, random, threading, time):
    shared_data = 0
    number_of_threads = 10000

    def update_shared_data():
        global shared_data

        temp = shared_data
        # Simulate some computation for each thread
        time.sleep(random.uniform(0,1)) 
        shared_data = temp + 1

    # Create number_of_threads to each increment the shared data by 1
    threads = [threading.Thread(target=update_shared_data) for _ in range(number_of_threads)]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print(f"Expected: {number_of_threads}\n  Actual: {shared_data}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **The Soloutions**

    To solve the issue of race conditions developers can use various synchronization techniques to co-ordintate threads such as mutexs, locks, conditions, semaphores and more, some of which we will explore now by adapting the unsafe code to a safe synchronized version using some of the listed techniques.


    **Mutexs**
    Mutexs ensure that a critical section of code is thread safe by defining a point in the code from which a thread must hold a lock (aquired from the mutex) to proceed to a critical section of code. Once a thread has aquired a lock and passed through the critical section of code it releases the lock for another thread to aquire before procceding. The exact form this mechanism can take varies across language implmentations, in some languages it's explicit such as in java where you must call .unlock() on the lock instance, in python however it can be handled implicitly using the "with" keyword as in the following example.
    *Note: In python the mutex class is called "Lock"*
    """)
    return


@app.cell
def _(print, random, threading, time):
    locked_shared_data = 0
    locked_number_of_threads = 10
    mutex = threading.Lock() # We create our lock instance

    def update_locked_shared_data():
        """
        We ensure that the critical section of code can only be executed by one thread (The one currently holding the lock)
        """
        global locked_shared_data

        with mutex: # Thread aquires lock before entering critical section
            temp = locked_shared_data
            time.sleep(random.uniform(0,1))
            locked_shared_data = (temp + 1)

    _threads = [threading.Thread(target=update_locked_shared_data) for _ in range(locked_number_of_threads)]
    for _thread in _threads:
        _thread.start()

    for _thread in _threads:
        _thread.join()

    print(f"Expected: {locked_number_of_threads}\n  Actual: {locked_shared_data}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Mutexes vs Synchronised Locking**

    Mutexs as already disscussed are a method for thread synchronization. Synchronized Locking is an higher level language abstraction/construct which also acheives mutal exclussion between threads, they are most commonly found in higher level languages notibly java with the synchronized key word and in C# with the lock key word.

    The difference between the two levels of abstraction is primarily the level of granularity with wich you control the lock and unlock mechanisms. When using mutexs you explicitly define the points at which a mutex is locked and unlocked. With synchronized locking you define a method/code block to be synchronized and the release of the lock when exiting the defined scope is handeled automatically even in the case of exceptions. However this is not the only difference between the two with Javas implmentation of synchronized for example using object association rather than OS mutexs under the hood to acheive mutual exclusion and mutexs in lower level languages offering more variants of locks such as timed-locks, read-write locks etc.

    In java the synchronized sections can be used with wait() and notify()/notifyAll() methods when finer control is required, were wait releasing the lock of the current thread and sendign it into a waiting state and were notify or notifyAll will wake up a/all waiting thread(s).

    In python there is not strictly any equivilant to synchronized locking as there is in Java.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Semaphores**
    Semaphores are another synchronization tool for coordinating multiple threads' access to shared resources. They differ from mutexes primarily in that semaphores signal between threads, while mutexes protect ownership of resources.

    With semaphores, threads either signal (post/increment) or wait (pend/decrement) to coordinate access. A counting semaphore can allow up to N threads to proceed concurrently where N is set at initalization of the semaphore. The current available count is incremented with signal() or decremented with wait() calls. When N threads are already active, additional threads wait until one signals completion.

    This can be usefull in the following applications:
    - When you wish to limit access to a finite number of identical resources
    - When you want to dynamically control throughput instead of serial execution


    *Further Reading:* [Mutexes and Semaphores Demystified](https://barrgroup.com/blog/mutexes-and-semaphores-demystified)

    The following program demonstrates the utility of a semaphore as we manage access of cars to a parking lot with only 3 spaces. Without the use of semaphores we see that the car parks capacity is exceeded and we let in more cars than there are spaces.

    *Note: We will use mutexs in both these examples to ensure that operations on global counters are atomic*
    """)
    return


@app.cell
def _(Console, mo, threading, time):
    _console = Console(record=True)

    # Shared resource: limited parking spots
    _available_spots = [3]
    _current_cars = [0]
    _race_condition = [False]
    _mutex = [threading.Lock(), threading.Lock(), threading.Lock()] 

    def park_car(car_id):

        time.sleep(0.1)  # Simulate parking time
        with _mutex[0]:
            _current_cars[0] += 1

        if _current_cars[0] > _available_spots[0]:
            with _mutex[2]:
                _race_condition[0] = True
            _console.print(f"Car {car_id} has entered car park, Cars in car park: {_current_cars[0]}, [red]Car couldn't park no space!!![/red]")

        else:
            _console.print(f"Car {car_id} has entered car park, Cars in car park: {_current_cars[0]}, [green]Car is parked[/green]")

        time.sleep(1)  # Car stays parked
        with _mutex[1]:
            _current_cars[0] -= 1


    # Create 10 cars trying to park
    _threads = []
    for _i in range(10):
        _t = threading.Thread(target=park_car, args=(_i,))
        _threads.append(_t)
        _t.start()

    for _t in _threads:
        _t.join()


    # Check if race condition has occurred
    if _race_condition[0]:
        _console.print("[red]FAILURE:[/red] More than 3 cars parked simultaneously!")
    else:
        _console.print("[green]SUCCESS:[/green] All cars parked successfully!")

    # Display output as html
    _html = _console.export_html(inline_styles=True)
    mo.Html(_html)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the next demo we resolve the issues of the previous cell through the use of a semaphore
    """)
    return


@app.cell
def _(Console, mo, threading, time):
    _console = Console(record=True)

    _available_spots = [3]
    _current_cars = [0]
    _race_condition = [False]
    _mutex = [threading.Lock(), threading.Lock(), threading.Lock()] 
    semaphore = threading.Semaphore(_available_spots[0]) # Create semaphore with count equal the number of availabe spots in the car park


    def _park_car(car_id):

        # Acquire a spot: decrement semaphore count, block if no spots available 
        semaphore.acquire()
        time.sleep(0.1)
        with _mutex[0]:
            _current_cars[0] += 1

        if _current_cars[0] > _available_spots[0]:
            with _mutex[2]:
                _race_condition[0] = True
            _console.print(f"Car {car_id} has entered car park, Cars in car park: {_current_cars[0]}, [red]Car couldn't park no space!!![/red]")
        else:
            _console.print(f"Car {car_id} has entered car park, Cars in car park: {_current_cars[0]}, [green]Car is parked[/green]")

        time.sleep(1)
        with _mutex[1]:
            _current_cars[0] -= 1   

        # Release a spot: increment semaphore count so another car can enter
        semaphore.release()


    _threads = []
    for _i in range(10):
        _t = threading.Thread(target=_park_car, args=(_i,))
        _threads.append(_t)
        _t.start()

    for _t in _threads:
        _t.join()

    if _race_condition[0]:
        _console.print("[red]FAILURE:[/red] More than 3 cars parked simultaneously!")
    else:
        _console.print("[green]SUCCESS:[/green] All cars parked successfully!")


    _html = _console.export_html(inline_styles=True)
    mo.Html(_html)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Multi-threading challanges

    **Deadlocks**
    Deadlocks are a common issue that can occur when designing a mult-threaded system. A deadlock occurs when a thread currently holding a lock cannot procced due to a dependency on a resource held by a thread currently waithing for the lock that this thread posses, essentially creating a circular dependency between the two locks.
    """)
    return


@app.cell
def _(Image):
    img = Image.open("images/deadlocks-image.jpg")
    img
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following python cell we will demonstrate how a deadlock may occur.
    Notice how the threads get stuck attempting to aquire the lock the other thread currently posseses.
    """)
    return


@app.cell
def _(Console, mo, threading, time):
    _console = Console(record=True)

    # Note we use lists as they are immutable
    _mutex_1 = [threading.Lock()]
    _mutex_2 = [threading.Lock()]

    def function_one():
        _console.print(f"{threading.current_thread().name} is [#FFA500]atempting[/#FFA500] to aquire the lock for _mutex_1 ...")
        _mutex_1[0].acquire()
        _console.print(f"{threading.current_thread().name} has [green]aquired[/green] _mutex_1")

        _console.print(f"{threading.current_thread().name} is [#FFA500]atempting[/#FFA500] to aquire the lock for _mutex_2 ...")
        time.sleep(0.5) # Pause thread proccess to ensure deadlock occurs

        timeout = _mutex_2[0].acquire(timeout=10)
        if timeout:
            _console.print(f"{threading.current_thread().name} has [green]aquired[/green] _mutex_1")
            _mutex_2[0].release()

        else:
            _console.print(f"{threading.current_thread().name} has [red]failed[/red] to aquire _mutex_1")

        time.sleep(5)
        _mutex_1[0].release()


    def function_two():
        _console.print(f"{threading.current_thread().name} is [#FFA500]attempting[/#FFA500] to aquire the lock for _mutex_2 ...")
        _mutex_2[0].acquire()
        _console.print(f"{threading.current_thread().name} has [green]aquired[/green] _mutex_2")


        _console.print(f"{threading.current_thread().name} is [#FFA500]atempting[/#FFA500] to aquire the lock for _mutex_1 ...")
        time.sleep(0.5) # Pause thread proccess to ensure deadlock occurs

        timeout = _mutex_1[0].acquire(timeout=10)
        if timeout:
            _console.print(f"{threading.current_thread().name} has [green]aquired[/green] _mutex_1")
            _mutex_1[0].release()

        else:
            _console.print(f"{threading.current_thread().name} has [red]failed[/red] to aquire _mutex_1")


        time.sleep(5)
        _mutex_2[0].release()


    _thread_1 = threading.Thread(name="Thread-1",target=function_one)
    _thread_2 = threading.Thread(name="Thread-2",target=function_two)

    _thread_1.start()
    _thread_2.start()
    _thread_1.join()
    _thread_2.join()
    _html = _console.export_html(inline_styles=True)
    mo.Html(_html)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Starvation**

    Starvation is another common issue encountered when designing multi-threaded systems. It is defined as a thread being perpetually deprived of resources it needs to execute, even though those resources are available. This issue can be caused in a number of different ways such as poor scheduling, resource hogging or as in the case of the demo below unfair locking i.e one thread repeatedly aquiring a mutex lock.
    """)
    return


@app.cell
def _(defaultdict, plt, print, threading, time):


    lock = threading.Lock()
    lock_history = []  # (timestamp, thread_name, event_type)
    start_time = time.time()

    def greedy_thread(name):
        """This thread hogs the lock"""
        count = 0
        while count < 50:  # Limited iterations for demo
            lock.acquire()
            lock_history.append((time.time() - start_time, name, "acquire"))
            time.sleep(0.01)  # Hold lock briefly
            lock_history.append((time.time() - start_time, name, "release"))
            lock.release()
            count += 1

    def starved_thread(name):
        """This thread gets starved"""
        attempt = 0
        while attempt < 10:
            lock.acquire()
            lock_history.append((time.time() - start_time, name, "acquire"))
            time.sleep(0.01)
            lock_history.append((time.time() - start_time, name, "release"))
            lock.release()
            attempt += 1
            time.sleep(0.05)

    # Start threads
    t1 = threading.Thread(target=greedy_thread, args=("Greedy-1",))
    t2 = threading.Thread(target=greedy_thread, args=("Greedy-2",))
    t3 = threading.Thread(target=starved_thread, args=("Starved",))

    t1.start()
    t2.start()
    time.sleep(0.1)  # Let greedy threads start
    t3.start()

    # Wait for all threads
    t1.join()
    t2.join()
    t3.join()

    # Process the history to calculate lock hold times
    thread_times = defaultdict(list)  # {thread_name: [(start, duration), ...]}
    thread_acquire_times = {}

    for timestamp, thread_name, event_type in lock_history:
        if event_type == "acquire":
            thread_acquire_times[thread_name] = timestamp
        elif event_type == "release" and thread_name in thread_acquire_times:
            start = thread_acquire_times[thread_name]
            duration = timestamp - start
            thread_times[thread_name].append((start, duration))
            del thread_acquire_times[thread_name]

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Timeline visualization
    colors = {"Greedy-1": "red", "Greedy-2": "orange", "Starved": "blue"}
    y_positions = {"Greedy-1": 2, "Greedy-2": 1, "Starved": 0}

    for thread_name, intervals in thread_times.items():
        for start, duration in intervals:
            ax1.barh(y_positions[thread_name], duration, left=start, 
                    height=0.8, color=colors[thread_name], alpha=0.7)

    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["Starved", "Greedy-2", "Greedy-1"])
    ax1.set_xlabel("Time (seconds)")
    ax1.set_title("Thread Lock Acquisition Timeline (Starvation Demo)")
    ax1.grid(axis='x', alpha=0.3)

    # Total time with lock (bar chart)
    total_times = {name: sum(d for _, d in intervals) 
                   for name, intervals in thread_times.items()}
    lock_counts = {name: len(intervals) for name, intervals in thread_times.items()}

    ax2.bar(total_times.keys(), total_times.values(), 
            color=[colors[name] for name in total_times.keys()], alpha=0.7)
    ax2.set_ylabel("Total Time Holding Lock (seconds)")
    ax2.set_title("Total Lock Hold Time by Thread")
    ax2.grid(axis='y', alpha=0.3)

    # Add count annotations
    for name, total in total_times.items():
        ax2.text(name, total, f'{lock_counts[name]} acquisitions', 
                ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print(f"\nLock Statistics:")
    for name in sorted(thread_times.keys()):
        print(f"{name}: {lock_counts[name]} acquisitions,")
        print(f"{total_times[name]:.3f}s total time with lock")
    return


if __name__ == "__main__":
    app.run()

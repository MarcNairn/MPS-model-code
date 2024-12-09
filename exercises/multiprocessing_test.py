import time
import math
from multiprocessing import Process, freeze_support, Pool, Queue, Manager


# Define a decorator to measure function execution time
def time_taken(func):
    """
    A decorator to measure the execution time of a function.
    
    Args:
        func: The target function.
    
    Returns:
        A wrapper function that measures and prints the execution time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Function {func.__name__!r} took {execution_time:.4f} seconds to execute.")
        return result
    return wrapper


# Define a compute-intensive task function
def compute_intensive_task(num):
    """
    A compute-intensive task that calculates the factorial of a number.
    
    Args:
        num: The input number.
    
    Returns:
        The factorial of the input number.
    """
    #return math.sqrt(num)  # This line is commented out to focus on threading
    return math.factorial(num)

# Define a compute-intensive task function
def compute_intensive_task2(procid, num, return_dict):
    """
    A compute-intensive task that calculates the factorial of a number.
    
    Args:
        num: The input number.
    
    Returns:
        The factorial of the input number.
    """
    #return math.sqrt(num)  # This line is commented out to focus on threading
    math.factorial(num)
    return_dict[procid] = procid


# Define single-threaded task function
@time_taken
def single_threaded_task(nums):
    """
    A single-threaded task that performs compute-intensive tasks sequentially.
    
    Args:
        nums: A list of input numbers.
    """
    for num in nums:
        compute_intensive_task(num)


# Define multi-processing task function
@time_taken
def multi_processing_task(nums):
    """
    A multi-processing task that creates and runs a process pool to perform 
    compute-intensive tasks concurrently.
    
    Args:
        nums: A list of input numbers.
    """
    processes = []
    manager = Manager()
    return_dict = manager.dict()

    # Create len(nums) processes
    for procid, num in enumerate(nums):
        process = Process(target=compute_intensive_task2, args=(procid, num, return_dict,))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    return return_dict


# Define multi-processing task function
@time_taken
def multi_processing_pooled(nums):
    """
    A multi-processing task that creates and runs a process pool to perform 
    compute-intensive tasks concurrently.
    
    Args:
        nums: A list of input numbers.
    """

    pool = Pool(processes=10)

    # Create len(nums) processes
    for num in nums:
        pool.apply_async(compute_intensive_task, args=(num,))
    pool.close()
    pool.join()


if __name__ == "__main__":
    #freeze_support()
    nums = [500100]*80

    # Run single-threaded task
    #single_threaded_task(nums)

    # Run multi-processing task
    return_dict = multi_processing_task(nums)
    print(return_dict.values())
    print([return_dict[val] for val in range(len(return_dict))])

    # Run multi-processing task with a process pool
    #multi_processing_pooled(nums)
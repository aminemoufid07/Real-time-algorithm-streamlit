import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# FCFS Scheduling
def fcfs_scheduling(tasks):
    tasks = [(task[0], task[1], task[2]) for task in tasks]
    tasks.sort(key=lambda x: x[1])  # Sort by arrival time
    current_time = 0
    gantt_chart = []
    waiting_times = {}

    for task in tasks:
        task_name, arrival_time, burst_time = task
        start_time = max(current_time, arrival_time)
        finish_time = start_time + burst_time
        gantt_chart.append((task_name, start_time, finish_time))
        waiting_time = start_time - arrival_time
        waiting_times[task_name] = waiting_time
        current_time = finish_time

    return gantt_chart, waiting_times




# Non-Preemptive SJF Scheduling
def non_preemptive_sjf(tasks):
    tasks = [(task[0], task[1], task[2]) for task in tasks]
    tasks.sort(key=lambda x: (x[1], x[2]))  # Sort by arrival time, then burst time
    current_time = 0
    gantt_chart = []
    waiting_times = {}
    task_completion_time = {}
    
    for task_name, arrival_time, burst_time in tasks:
        start_time = max(current_time, arrival_time)
        end_time = start_time + burst_time
        gantt_chart.append((task_name, start_time, end_time))
        task_completion_time[task_name] = end_time
        waiting_times[task_name] = start_time - arrival_time
        current_time = end_time
    
    average_waiting_time = sum(waiting_times.values()) / len(waiting_times)
    return gantt_chart, waiting_times, average_waiting_time


# Preemptive SJF Scheduling
def preemptive_sjf(tasks):
    tasks = [(task[0], task[1], task[2]) for task in tasks]
    tasks.sort(key=lambda x: (x[1], x[2]))  # Sort by arrival time, then burst time
    current_time = 0
    gantt_chart = []
    waiting_times = {}
    remaining_burst = {task[0]: task[2] for task in tasks}
    task_completion_time = {}
    task_start_time = {task[0]: None for task in tasks}  # Track when tasks start execution
    task_wait_time = {task[0]: 0 for task in tasks}  # Track total waiting time of each task
    task_end_time = {task[0]: 0 for task in tasks}  # Track end times of tasks

    # Preemptive SJF: Loop until all tasks are completed
    while tasks or any(remaining_burst.values()):
        # Get all tasks that have arrived
        available_tasks = [task for task in tasks if task[1] <= current_time]
        
        if not available_tasks:
            current_time += 1
            continue

        # Select the task with the smallest burst time that is ready to execute
        next_task = min(available_tasks, key=lambda x: remaining_burst[x[0]])

        task_name, arrival_time, burst_time = next_task

        # If it's the first time the task is being processed, record the start time
        if task_start_time[task_name] is None:
            task_start_time[task_name] = current_time

        # Process the task for 1 time unit
        gantt_chart.append((task_name, current_time, current_time + 1))
        remaining_burst[task_name] -= 1

        # Increment waiting time for the task
        if remaining_burst[task_name] > 0:  # Task is waiting before it's completed
            task_wait_time[task_name] += 1

        current_time += 1

        # Remove completed tasks
        if remaining_burst[task_name] == 0:
            tasks = [task for task in tasks if task[0] != task_name]
            task_end_time[task_name] = current_time

            # Calculate waiting time: total time in system - burst time
            total_time_in_system = task_end_time[task_name] - arrival_time
            waiting_times[task_name] = total_time_in_system - burst_time

    # Calculate average waiting time
    average_waiting_time = sum(waiting_times.values()) / len(waiting_times)

    return gantt_chart, waiting_times, average_waiting_time




def rate_monotonic_scheduling(tasks, hyper_period=20):
    """
    Rate Monotonic Scheduling with fixed Gantt chart handling.
    """
    tasks.sort(key=lambda x: x[2])  # Sort tasks by period (ascending)
    current_time = 0
    gantt_chart = []
    remaining_burst = {task[0]: task[1] for task in tasks}  # Remaining burst time for each task
    deadlines = {task[0]: task[2] for task in tasks}  # Deadlines for each task (period)
    waiting_times = {task[0]: 0 for task in tasks}  # Waiting time for each task
    task_start_time = {task[0]: None for task in tasks}  # Track start times of tasks
    task_finish_time = {task[0]: None for task in tasks}  # Track finish times of tasks
    missed_deadlines = []

    while current_time < hyper_period:
        # Refresh tasks at the start of their periods
        for task_name, burst_time, period in tasks:
            if current_time % period == 0:
                if remaining_burst[task_name] > 0:  # Missed deadline
                    missed_deadlines.append((task_name, current_time))
                remaining_burst[task_name] = burst_time  # Reset burst time for the task
                deadlines[task_name] = current_time + period  # Update deadline

        # Select the task with the shortest period (highest priority)
        ready_tasks = [
            task for task in tasks if remaining_burst[task[0]] > 0 and current_time < deadlines[task[0]]
        ]

        if ready_tasks:
            # Execute the highest-priority task (task with the shortest period)
            next_task = ready_tasks[0]
            task_name = next_task[0]

            # If it's the first time this task is starting, record the start time
            if task_start_time[task_name] is None:
                task_start_time[task_name] = current_time

            # Add this task to the gantt chart
            gantt_chart.append((task_name, current_time, current_time + 1))

            # Decrement remaining burst time of the task
            remaining_burst[task_name] -= 1

            # If task finishes, record finish time
            if remaining_burst[task_name] == 0:
                task_finish_time[task_name] = current_time + 1
        else:
            # Idle time if no tasks are ready
            gantt_chart.append(("Idle", current_time, current_time + 1))

        # Update waiting times for tasks that are ready to run
        for task_name in remaining_burst:
            if remaining_burst[task_name] > 0 and current_time < deadlines[task_name]:
                waiting_times[task_name] += 1

        current_time += 1

    # Calculate the final waiting time for each task
    for task_name in tasks:
        # If the task was executed at least once, calculate the waiting time
        if task_start_time[task_name[0]] is not None and task_finish_time[task_name[0]] is not None:
            total_time_in_system = task_finish_time[task_name[0]] - task_start_time[task_name[0]]
            preemptive_time = task_finish_time[task_name[0]] - task_start_time[task_name[0]]
            waiting_times[task_name[0]] = (
                task_start_time[task_name[0]] - task_name[1] + (total_time_in_system - preemptive_time)
            )

    return gantt_chart, waiting_times






# Streamlit UI
st.title("Scheduling Algorithm Simulation")
st.write("Choose an algorithm and input tasks to see the Gantt chart and performance metrics.")

algorithm = st.selectbox(
    "Select Scheduling Algorithm",
    ["FCFS (First Come First Serve)", "SJF Preemptive", "SJF Non-Preemptive", "Rate Monotonic"]
)

num_tasks = st.number_input("Number of tasks", min_value=1, max_value=10, value=3, step=1)

if "tasks" not in st.session_state or len(st.session_state.tasks) != num_tasks:
    st.session_state.tasks = [{"name": f"Task {i+1}", "arrival_time": 0, "burst_time": 1} for i in range(num_tasks)]

for i, task in enumerate(st.session_state.tasks):
    cols = st.columns(4)
    with cols[0]:
        task["name"] = st.text_input(f"Name of Task {i + 1}", value=task["name"], key=f"name_{i}")
    with cols[1]:
        task["arrival_time"] = st.number_input(f"Arrival Time of Task {i + 1}", min_value=0, value=task["arrival_time"], key=f"arrival_{i}")
    with cols[2]:
        task["burst_time"] = st.number_input(f"Burst Time of Task {i + 1}", min_value=1, value=task["burst_time"], key=f"burst_{i}")
    with cols[3]:
        task["period"] = st.number_input(f"Period of Task {i + 1}", min_value=1, value=task.get("period", 10), key=f"period_{i}")

tasks = [ (task["name"], task["arrival_time"], task["burst_time"], task["period"])
    for task in st.session_state.tasks]

# Determine hyperperiod
import math
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

hyperperiod = math.prod([task[3] for task in tasks])




if st.button("Simulate"):
    gantt_chart = []  # Initialize to avoid NameError
    waiting_times = {}  # Initialize to handle all algorithms
    avg_waiting_time = 0  # Default for algorithms with average waiting time
    missed_deadlines = []  # Default for Rate Monotonic Scheduling

    if algorithm == "FCFS (First Come First Serve)":
        gantt_chart, waiting_times = fcfs_scheduling(tasks)
        avg_waiting_time = sum(waiting_times.values()) / len(waiting_times)

    elif algorithm == "SJF Preemptive":
        gantt_chart, waiting_times, avg_waiting_time = preemptive_sjf(tasks)

    elif algorithm == "SJF Non-Preemptive":
        gantt_chart, waiting_times, avg_waiting_time = non_preemptive_sjf(tasks)

    elif algorithm == "Rate Monotonic":
        tasks_with_periods = [(task["name"], task["burst_time"], task["period"]) for task in st.session_state.tasks]
        gantt_chart, waiting_times = rate_monotonic_scheduling(tasks_with_periods, hyper_period=20)

    # Assign a unique color for each task
    task_colors = {task[0]: plt.cm.tab10(i) for i, task in enumerate(tasks)}

    # Combined Individual Diagrams
    st.subheader("Individual Task Execution Timeline")
    fig, ax = plt.subplots(figsize=(10, len(tasks) * 1.5))

    # Set a unique color for each task
    task_colors = {task_name: plt.cm.tab10(i) for i, task_name in enumerate(set([task[0] for task in gantt_chart]))}

    # Plot individual task execution timeline
    for task_name in task_colors.keys():
        for segment in [t for t in gantt_chart if t[0] == task_name]:
            _, start_time, end_time = segment
            ax.barh(task_name, end_time - start_time, left=start_time, color=task_colors[task_name], edgecolor="black")
            ax.text((start_time + end_time) / 2, task_name, f"{end_time - start_time}", ha="center", va="center", color="white")

    ax.set_xlim(0, sum([task[2] for task in tasks]) + 5)  # Adjust x-axis limit for clarity
    ax.set_xlabel("Time")
    ax.set_ylabel("Tasks")
    ax.set_title("Execution Timeline for Individual Tasks")
    st.pyplot(fig)

    # Gantt Chart
    st.subheader("Gantt Chart")
    fig, ax = plt.subplots(figsize=(10, 4))

    # Merge consecutive identical tasks and sum their durations
    merged_blocks = []
    current_task = None
    current_start = None
    current_end = None

    for task_name, start_time, end_time in gantt_chart:
        if task_name == current_task:
            current_end = end_time
        else:
            if current_task is not None:
                merged_blocks.append((current_task, current_start, current_end))
            current_task = task_name
            current_start = start_time
            current_end = end_time

    if current_task is not None:
        merged_blocks.append((current_task, current_start, current_end))

    for task_name, start_time, end_time in merged_blocks:
        ax.barh("Tasks", end_time - start_time, left=start_time, color=task_colors[task_name], edgecolor="black")
        ax.text((start_time + end_time) / 2, 0, str(end_time - start_time), ha="center", va="center", color="white")

    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart")
    st.pyplot(fig)

    # Waiting Times and Results
    st.subheader("Waiting Times")
    results = pd.DataFrame({
        "Task": [task[0] for task in tasks],
        "Arrival Time": [task[1] for task in tasks],
        "Burst Time": [task[2] for task in tasks],
        "Waiting Time": [waiting_times[task[0]] for task in tasks],
    })
    st.table(results)

    # Average Waiting Time
    st.subheader("Average Waiting Time")
    st.write(f"The average waiting time is: {avg_waiting_time:.2f}")

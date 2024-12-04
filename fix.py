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
    
    while tasks or any(remaining_burst.values()):
        available_tasks = [task for task in tasks if task[1] <= current_time]

        if not available_tasks:
            current_time += 1
            continue

        next_task = min(available_tasks, key=lambda x: remaining_burst[x[0]])

        task_name, arrival_time, burst_time = next_task

        # Process the task for 1 time unit
        gantt_chart.append((task_name, current_time, current_time + 1))
        remaining_burst[task_name] -= 1
        current_time += 1

        # Remove completed tasks
        if remaining_burst[task_name] == 0:
            tasks = [task for task in tasks if task[0] != task_name]
            task_completion_time[task_name] = current_time
            waiting_times[task_name] = (
                current_time - arrival_time + burst_time
            )  # Waiting time = time in system - burst time

    average_waiting_time = sum(waiting_times.values()) / len(waiting_times)
    return gantt_chart, waiting_times, average_waiting_time

# Rate Monotonic Scheduling
def rate_monotonic_scheduling(tasks, hyper_period=20):
    """
    Rate Monotonic Scheduling with fixed Gantt chart handling.
    """
    tasks.sort(key=lambda x: x[2])  # Sort tasks by period
    current_time = 0
    gantt_chart = []
    remaining_burst = {task[0]: 0 for task in tasks}
    deadlines = {task[0]: task[2] for task in tasks}
    waiting_times = {task[0]: 0 for task in tasks}
    missed_deadlines = []

    while current_time < hyper_period:
        # Refresh tasks at the start of their periods
        for task_name, burst_time, period in tasks:
            if current_time % period == 0:
                if remaining_burst[task_name] > 0:  # Deadline missed
                    missed_deadlines.append((task_name, current_time))
                remaining_burst[task_name] = burst_time  # Reset burst time
                deadlines[task_name] = current_time + period  # Update deadline

        # Select the task with the shortest period (highest priority)
        ready_tasks = [
            task for task in tasks if remaining_burst[task[0]] > 0 and current_time < deadlines[task[0]]
        ]

        if ready_tasks:
            # Execute the highest-priority task
            next_task = ready_tasks[0]
            task_name = next_task[0]
            gantt_chart.append((task_name, current_time, current_time + 1))
            remaining_burst[task_name] -= 1
        else:
            # Idle time
            gantt_chart.append(("Idle", current_time, current_time + 1))

        # Update waiting times for all ready tasks
        for task_name in remaining_burst:
            if remaining_burst[task_name] > 0 and current_time < deadlines[task_name]:
                waiting_times[task_name] += 1

        current_time += 1

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
        gantt_chart, waiting_times,  avg_waiting_time = preemptive_sjf(tasks)
    elif algorithm == "SJF Non-Preemptive":
        gantt_chart, waiting_times, avg_waiting_time = non_preemptive_sjf(tasks)
    elif algorithm == "Rate Monotonic":
        tasks_with_periods = [(task["name"], task["burst_time"], task["period"]) for task in st.session_state.tasks]
        gantt_chart, waiting_times = rate_monotonic_scheduling(tasks_with_periods, hyper_period=20)


    
    if algorithm != "Rate Monotonic":
     st.subheader("Waiting Times")
     waiting_time_df = pd.DataFrame({
        "Task": waiting_times.keys(),
        "Waiting Time": waiting_times.values(),
    })
     st.table(waiting_time_df)

     st.subheader("Average Waiting Time")
     st.write(f"The average waiting time is: {avg_waiting_time:.2f} units")

    # else:
    #     # Missed Deadlines for Rate Monotonic
    #     st.subheader("Missed Deadlines")
    #     if missed_deadlines:
    #         st.write("The following deadlines were missed:")
    #         st.table(pd.DataFrame(missed_deadlines, columns=["Task", "Time"]))
    #     else:
    #         st.write("No deadlines were missed.")

    # Assign a unique color for each task
    task_colors = {task[0]: plt.cm.tab10(i) for i, task in enumerate(tasks)}

    # Combined Individual Diagrams
    st.subheader("Individual Task Execution Timeline")
    fig, ax = plt.subplots(figsize=(10, len(tasks) * 1.5))
    task_names = {task[0] for task in gantt_chart}  # Unique task names
    for i, task_name in enumerate(task_names):
        for segment in [t for t in gantt_chart if t[0] == task_name]:  # Find all segments for the task
            _, start_time, end_time = segment
            ax.barh(task_name, end_time - start_time, left=start_time, color=task_colors.get(task_name, "gray"), edgecolor="black")
            ax.text((start_time + end_time) / 2, i, f"{end_time - start_time}", ha="center", va="center", color="white")

    ax.set_xlim(0, sum([task[2] for task in tasks]) + 5)  # Adjust x-axis limit for clarity
    ax.set_xlabel("Time")
    ax.set_ylabel("Tasks")
    ax.set_title("Execution Timeline for Individual Tasks")
    st.pyplot(fig)

    # Gantt Chart
    st.subheader("Gantt Chart")
    fig, ax = plt.subplots(figsize=(10, 4))
    for task_name, start_time, end_time in gantt_chart:
        ax.barh("Tasks", end_time - start_time, left=start_time, color=task_colors.get(task_name, "gray"), edgecolor="black", label=task_name)
        ax.text((start_time + end_time) / 2, 0, task_name, ha="center", va="center", color="white")
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart")
    ax.legend()
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

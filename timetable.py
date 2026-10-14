import csv
import random
import time


# ============================================================
# PROBLEM 4: EXAM TIMETABLE SCHEDULING
# Algorithms:
# 1. Welsh-Powell Greedy
# 2. Exact Backtracking with DSatur selection
# ============================================================


def read_manual_input():
    """Read the conflict graph from user input."""
    n, m, r = map(
        int,
        input("Enter N (courses), M (conflicts), R (max exams per slot): ").split()
    )

    if n < 1 or r < 1 or m < 0:
        raise ValueError("N and R must be positive; M cannot be negative.")

    graph = [set() for _ in range(n)]

    print(f"Enter {m} conflict edges (u v), using course numbers 1 to {n}:")

    for _ in range(m):
        u, v = map(int, input().split())

        if not (1 <= u <= n and 1 <= v <= n) or u == v:
            raise ValueError(f"Invalid edge: {u} {v}")

        graph[u - 1].add(v - 1)
        graph[v - 1].add(u - 1)

    return graph, r


def generate_random_graph():
    """Generate a random conflict graph using edge density."""
    n, r = map(
        int,
        input("Enter N (courses) and R (max exams per slot): ").split()
    )

    if n < 1 or r < 1:
        raise ValueError("N and R must be positive.")

    density = float(input("Enter edge density from 0 to 1 (e.g. 0.35): "))

    if not 0 <= density <= 1:
        raise ValueError("Edge density must be between 0 and 1.")

    graph = [set() for _ in range(n)]

    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < density:
                graph[u].add(v)
                graph[v].add(u)

    print("Random conflict graph generated.")
    return graph, r


def validate_graph(graph, r):
    """Check whether the hall limit makes a timetable possible."""
    n = len(graph)

    # Each course needs one exam position, and a slot has R positions.
    if r < 1:
        raise ValueError("R must be at least 1.")

    if n == 0:
        raise ValueError("At least one course is required.")


def can_place(course, slot, assignment, slot_sizes, graph, r):
    """Check conflict and capacity constraints."""
    if slot_sizes[slot] >= r:
        return False

    for other in graph[course]:
        if assignment[other] == slot:
            return False

    return True


def welsh_powell(graph, r):
    """Greedy Welsh-Powell graph colouring with a hall limit."""
    start = time.perf_counter()
    n = len(graph)

    # Highest-degree courses are scheduled first.
    order = sorted(
        range(n),
        key=lambda course: len(graph[course]),
        reverse=True
    )

    assignment = [-1] * n
    slot_sizes = []
    slots_used = 0

    for course in order:
        placed = False

        for slot in range(slots_used):
            if can_place(course, slot, assignment, slot_sizes, graph, r):
                assignment[course] = slot
                slot_sizes[slot] += 1
                placed = True
                break

        if not placed:
            assignment[course] = slots_used
            slot_sizes.append(1)
            slots_used += 1

    elapsed = time.perf_counter() - start
    return assignment, slots_used, elapsed


def greedy_clique_lower_bound(graph, r):
    """
    Construct a valid clique as a lower bound.
    A clique's courses must all use different slots.
    Also apply the capacity lower bound ceil(N/R).
    """
    n = len(graph)
    order = sorted(
        range(n),
        key=lambda course: len(graph[course]),
        reverse=True
    )

    clique = []

    for course in order:
        if all(member in graph[course] for member in clique):
            clique.append(course)

    capacity_bound = (n + r - 1) // r
    return max(1, len(clique), capacity_bound)


def choose_dsatur_course(graph, assignment):
    """Choose an unassigned course using the DSatur heuristic."""
    unassigned = [
        course for course in range(len(graph))
        if assignment[course] == -1
    ]

    def priority(course):
        neighbor_slots = {
            assignment[neighbor]
            for neighbor in graph[course]
            if assignment[neighbor] != -1
        }

        return (
            len(neighbor_slots),
            len(graph[course]),
            -course
        )

    return max(unassigned, key=priority) if unassigned else None


def exact_backtracking(graph, r, initial_assignment, initial_slots):
    """
    Exact solver:
    Try successively smaller slot limits.
    DSatur selects the next course.
    If a solution is found for a limit, it is feasible.
    Exhausting all smaller limits proves optimality.
    """
    start = time.perf_counter()
    n = len(graph)

    best_assignment = initial_assignment[:]
    best_slots = initial_slots
    nodes_searched = 0

    lower_bound = greedy_clique_lower_bound(graph, r)

    def search(assignment, slot_sizes, limit, assigned_count):
        nonlocal nodes_searched
        nodes_searched += 1

        if assigned_count == n:
            return assignment[:]

        course = choose_dsatur_course(graph, assignment)

        # Only use existing slots or introduce the next new slot.
        highest_used = max(
            (value for value in assignment if value != -1),
            default=-1
        )

        max_slot_to_try = min(highest_used + 1, limit - 1)

        for slot in range(max_slot_to_try + 1):
            if not can_place(
                course, slot, assignment, slot_sizes, graph, r
            ):
                continue

            assignment[course] = slot
            slot_sizes[slot] += 1

            result = search(
                assignment, slot_sizes, limit, assigned_count + 1
            )

            if result is not None:
                return result

            assignment[course] = -1
            slot_sizes[slot] -= 1

        return None

    # Test every slot count below the greedy solution.
    for limit in range(lower_bound, best_slots):
        assignment = [-1] * n
        slot_sizes = [0] * limit

        result = search(assignment, slot_sizes, limit, 0)

        if result is not None:
            best_assignment = result
            best_slots = limit
            break

    elapsed = time.perf_counter() - start

    return (
        best_assignment,
        best_slots,
        elapsed,
        nodes_searched,
        True
    )


def make_timetable(assignment, slot_count):
    """Convert course assignments into a slot-wise timetable."""
    timetable = [[] for _ in range(slot_count)]

    for course, slot in enumerate(assignment):
        timetable[slot].append(course + 1)

    return timetable


def verify_timetable(graph, assignment, r):
    """Verify conflicts and hall capacity."""
    if any(slot < 0 for slot in assignment):
        return False

    timetable = make_timetable(
        assignment, max(assignment, default=-1) + 1
    )

    for courses in timetable:
        if len(courses) > r:
            return False

    for course, neighbors in enumerate(graph):
        for neighbor in neighbors:
            if assignment[course] == assignment[neighbor]:
                return False

    return True


def print_result(title, assignment, slot_count, elapsed, optimal):
    print(f"\n{'=' * 55}")
    print(title)
    print(f"{'=' * 55}")
    print("Number of slots:", slot_count)
    print("Optimality proven:", "Yes" if optimal else "No")
    print(f"Execution time: {elapsed:.6f} seconds")

    timetable = make_timetable(assignment, slot_count)

    for index, courses in enumerate(timetable, start=1):
        print(f"Slot {index}: {', '.join(map(str, courses))}")


def export_csv(greedy_assignment, greedy_slots,
               exact_assignment, exact_slots, r):
    """
    Export both schedules.
    Each row is one slot; columns are exam positions/halls.
    """
    max_slots = max(greedy_slots, exact_slots)

    filename = "exam_timetable.csv"

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Algorithm",
            "Slot",
            *[f"Hall {i}" for i in range(1, r + 1)]
        ])

        for name, assignment, slot_count in [
            ("Welsh-Powell", greedy_assignment, greedy_slots),
            ("Exact Backtracking", exact_assignment, exact_slots)
        ]:
            timetable = make_timetable(assignment, slot_count)

            for slot_index, courses in enumerate(timetable, start=1):
                row = [name, slot_index] + courses

                # Keep exactly R hall columns.
                row = row[:2 + r]
                row += [""] * (2 + r - len(row))

                writer.writerow(row)

    print(f"\nCSV file created: {filename}")


def main():
    print("=" * 55)
    print("EXAM TIMETABLE SCHEDULING - GRAPH COLOURING")
    print("=" * 55)
    print("1. Enter conflict graph manually")
    print("2. Generate random conflict graph")

    option = input("Choose option (1 or 2): ").strip()

    if option == "1":
        graph, r = read_manual_input()
    elif option == "2":
        graph, r = generate_random_graph()
    else:
        print("Invalid option.")
        return

    validate_graph(graph, r)

    print("\nCourses:", len(graph))
    print("Maximum exams per slot (R):", r)
    print("Conflicts:", sum(len(row) for row in graph) // 2)

    # Algorithm 1: greedy
    greedy_assignment, greedy_slots, greedy_time = welsh_powell(graph, r)

    # Algorithm 2: exact backtracking
    (
        exact_assignment,
        exact_slots,
        exact_time,
        nodes_searched,
        optimal
    ) = exact_backtracking(
        graph, r, greedy_assignment, greedy_slots
    )

    # Verify the schedules before printing/exporting.
    if not verify_timetable(graph, greedy_assignment, r):
        raise RuntimeError("Greedy timetable failed verification.")

    if not verify_timetable(graph, exact_assignment, r):
        raise RuntimeError("Exact timetable failed verification.")

    print_result(
        "WELSH-POWELL GREEDY RESULT",
        greedy_assignment,
        greedy_slots,
        greedy_time,
        False
    )

    print_result(
        "EXACT BACKTRACKING RESULT",
        exact_assignment,
        exact_slots,
        exact_time,
        optimal
    )

    print("\n" + "=" * 55)
    print("ALGORITHM COMPARISON")
    print("=" * 55)
    print(f"Welsh-Powell slots: {greedy_slots}")
    print(f"Exact minimum slots: {exact_slots}")
    print(f"Slots saved by exact method: {greedy_slots - exact_slots}")
    print("Greedy optimality proven: No")
    print("Exact optimality proven:", "Yes" if optimal else "No")
    print("Exact search nodes visited:", nodes_searched)

    export_csv(
        greedy_assignment, greedy_slots,
        exact_assignment, exact_slots, r
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print("Input error:", error)
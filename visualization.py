import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import math
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import to_rgba


# ============================================================
# EXAM TIMETABLE VISUALIZATION DASHBOARD
# Welsh-Powell + Exact Backtracking + Animation + Charts
# ============================================================

COLORS = [
    "#3182F6", "#FF9635", "#35B86B", "#F0444A",
    "#FFD52A", "#8A50D8", "#27B8B0", "#E66BA7",
    "#8C9BAA", "#B38B52", "#6979E8", "#9CCB42"
]

BG = "#F4F8FD"
NAVY = "#19416F"
TEXT = "#173454"

# ------------------ INPUT ------------------

def read_graph():
    print("\n=== EXAM TIMETABLE VISUALIZATION ===")
    print("1. Enter conflict graph manually")
    print("2. Generate a random conflict graph")

    option = input("Choose option: ").strip()

    if option == "1":
        n, m, r = map(
            int,
            input("Enter N M R: ").split()
        )

        if n < 1 or m < 0 or r < 1:
            raise ValueError("N and R must be positive; M cannot be negative.")

        graph = nx.Graph()
        graph.add_nodes_from(range(1, n + 1))

        print("Enter each conflict edge as: u v")

        for _ in range(m):
            u, v = map(int, input().split())

            if u not in graph or v not in graph or u == v:
                raise ValueError("Invalid conflict edge.")

            graph.add_edge(u, v)

        return graph, r

    if option == "2":
        n, r = map(
            int,
            input("Enter N and R: ").split()
        )

        density = float(input("Enter edge density (0 to 1): "))

        if n < 1 or r < 1 or not 0 <= density <= 1:
            raise ValueError("Invalid N, R, or edge density.")

        graph = nx.Graph()
        graph.add_nodes_from(range(1, n + 1))

        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                if random.random() < density:
                    graph.add_edge(u, v)

        print("Random conflict graph generated.")
        return graph, r

    raise ValueError("Please choose option 1 or 2.")


# ------------------ GREEDY ALGORITHM ------------------

def greedy_coloring(graph, r):
    start = time.perf_counter()

    # Welsh-Powell: highest-degree course first.
    order = sorted(
        graph.nodes(),
        key=lambda node: graph.degree(node),
        reverse=True
    )

    assignment = {}
    slot_sizes = []

    for course in order:
        placed = False

        for slot in range(len(slot_sizes)):
            if slot_sizes[slot] >= r:
                continue

            conflict_found = any(
                assignment.get(neighbor) == slot
                for neighbor in graph.neighbors(course)
            )

            if not conflict_found:
                assignment[course] = slot
                slot_sizes[slot] += 1
                placed = True
                break

        if not placed:
            assignment[course] = len(slot_sizes)
            slot_sizes.append(1)

    elapsed = time.perf_counter() - start
    return assignment, len(slot_sizes), elapsed


# ------------------ EXACT ALGORITHM ------------------

def exact_coloring(graph, r, initial_assignment, initial_slots):
    """
    Exact backtracking with DSatur course selection.
    Searches slot limits below the greedy solution.
    """
    start = time.perf_counter()
    nodes = list(graph.nodes())
    n = len(nodes)

    best_assignment = initial_assignment.copy()
    best_slots = initial_slots
    search_nodes = 0

    if n == 0:
        return {}, 0, 0.0, 0, True

    # A valid lower bound from capacity and a constructed clique.
    capacity_bound = math.ceil(n / r)

    order = sorted(
        nodes,
        key=lambda node: graph.degree(node),
        reverse=True
    )
    clique = []

    for course in order:
        if all(graph.has_edge(course, member) for member in clique):
            clique.append(course)

    lower_bound = max(1, capacity_bound, len(clique))

    def choose_course(assignment):
        unassigned = [
            course for course in nodes
            if course not in assignment
        ]

        def priority(course):
            neighbor_slots = {
                assignment[neighbor]
                for neighbor in graph.neighbors(course)
                if neighbor in assignment
            }
            return (
                len(neighbor_slots),
                graph.degree(course),
                -course
            )

        return max(unassigned, key=priority) if unassigned else None

    def search(assignment, sizes, limit):
        nonlocal search_nodes
        search_nodes += 1

        if len(assignment) == n:
            return assignment.copy()

        course = choose_course(assignment)

        # Symmetry breaking: only open the next unused slot.
        used_slots = len(sizes)
        max_slot = min(used_slots, limit - 1)

        for slot in range(max_slot + 1):
            is_new = slot == used_slots

            if is_new:
                sizes.append(0)

            if sizes[slot] >= r:
                if is_new:
                    sizes.pop()
                continue

            if any(
                assignment.get(neighbor) == slot
                for neighbor in graph.neighbors(course)
            ):
                if is_new:
                    sizes.pop()
                continue

            assignment[course] = slot
            sizes[slot] += 1

            result = search(assignment, sizes, limit)

            if result is not None:
                return result

            del assignment[course]
            sizes[slot] -= 1

            if is_new:
                sizes.pop()

        return None

    for limit in range(lower_bound, best_slots):
        result = search({}, [], limit)

        if result is not None:
            best_assignment = result
            best_slots = limit
            break

    elapsed = time.perf_counter() - start
    return best_assignment, best_slots, elapsed, search_nodes, True


# ------------------ VALIDATION ------------------

def verify(graph, assignment, r):
    if set(assignment) != set(graph.nodes()):
        return False

    counts = {}

    for course, slot in assignment.items():
        counts[slot] = counts.get(slot, 0) + 1

        if counts[slot] > r:
            return False

        for neighbor in graph.neighbors(course):
            if assignment.get(neighbor) == slot:
                return False

    return True


# ------------------ DASHBOARD ------------------

class TimetableDashboard:
    def __init__(self, root, graph, r):
        self.root = root
        self.graph = graph
        self.r = r

        self.greedy_assignment, self.greedy_slots, self.greedy_time = (
            greedy_coloring(graph, r)
        )

        (
            self.exact_assignment,
            self.exact_slots,
            self.exact_time,
            self.search_nodes,
            self.optimal
        ) = exact_coloring(
            graph, r,
            self.greedy_assignment,
            self.greedy_slots
        )

        if not verify(graph, self.greedy_assignment, r):
            raise ValueError("Greedy timetable failed validation.")

        if not verify(graph, self.exact_assignment, r):
            raise ValueError("Exact timetable failed validation.")

        self.assignment = {}
        self.animation_order = sorted(
            graph.nodes(),
            key=lambda node: graph.degree(node),
            reverse=True
        )

        self.animation_index = 0
        self.animation_running = False
        self.after_id = None

        self.root.title("Exam Timetable Scheduling Dashboard")
        self.root.geometry("1450x900")
        self.root.configure(bg=BG)

        self.build_ui()
        self.draw_all()

    def build_ui(self):
        header = tk.Frame(self.root, bg=NAVY, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="EXAM TIMETABLE SCHEDULING",
            bg=NAVY,
            fg="white",
            font=("Arial", 22, "bold")
        ).pack(side="left", padx=25, pady=16)

        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=12, pady=8)

        ttk.Button(
            controls, text="Show Final Timetable",
            command=self.show_final
        ).pack(side="left", padx=5)

        ttk.Button(
            controls, text="Start Animation",
            command=self.start_animation
        ).pack(side="left", padx=5)

        ttk.Button(
            controls, text="Pause",
            command=self.pause_animation
        ).pack(side="left", padx=5)

        ttk.Button(
            controls, text="Reset",
            command=self.reset_animation
        ).pack(side="left", padx=5)

        ttk.Button(
            controls, text="Export CSV",
            command=self.export_csv
        ).pack(side="left", padx=5)

        self.status = tk.Label(
            self.root,
            text="",
            bg=BG,
            fg=NAVY,
            font=("Arial", 12, "bold")
        )
        self.status.pack(fill="x", padx=20, pady=3)

        # Two top panels
        self.top = tk.Frame(self.root, bg=BG)
        self.top.pack(fill="both", expand=True, padx=10)

        self.graph_panel = tk.LabelFrame(
            self.top,
            text="Conflict Graph",
            bg="white",
            fg=NAVY,
            font=("Arial", 13, "bold")
        )
        self.graph_panel.pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )

        self.table_panel = tk.LabelFrame(
            self.top,
            text="Final Timetable Grid",
            bg="white",
            fg=NAVY,
            font=("Arial", 13, "bold")
        )
        self.table_panel.pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )

        # Bottom panels
        self.bottom = tk.Frame(self.root, bg=BG)
        self.bottom.pack(fill="both", expand=True, padx=10)

        self.animation_panel = tk.LabelFrame(
            self.bottom,
            text="Assignment Animation",
            bg="white",
            fg=NAVY,
            font=("Arial", 13, "bold")
        )
        self.animation_panel.pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )

        self.chart_panel = tk.LabelFrame(
            self.bottom,
            text="Algorithm Comparison as N Increases",
            bg="white",
            fg=NAVY,
            font=("Arial", 13, "bold")
        )
        self.chart_panel.pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )

    def color_for_slot(self, slot):
        return COLORS[slot % len(COLORS)]

    def draw_all(self):
        self.draw_graph()
        self.draw_timetable()
        self.draw_animation_graph()
        self.draw_comparison()

    def draw_graph(self):
        for child in self.graph_panel.winfo_children():
            child.destroy()

        fig = Figure(figsize=(5.4, 3.4), dpi=95)
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        pos = nx.spring_layout(
            self.graph, seed=42, k=1.1
        )

        assignment = self.assignment or self.exact_assignment

        node_colors = [
            self.color_for_slot(assignment[node])
            for node in self.graph.nodes()
        ]

        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            edge_color="#6B7785", width=1.5
        )

        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors,
            node_size=650,
            edgecolors="#26384A",
            linewidths=1.2
        )

        nx.draw_networkx_labels(
            self.graph, pos, ax=ax,
            font_size=10,
            font_weight="bold"
        )

        ax.set_title(
            f"Courses: {len(self.graph.nodes())}   "
            f"Conflicts: {self.graph.number_of_edges()}   "
            f"R: {self.r}",
            fontsize=10
        )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.graph_canvas = canvas

    def draw_timetable(self):
        for child in self.table_panel.winfo_children():
            child.destroy()

        assignment = self.assignment or self.exact_assignment
        slot_count = (
            max(assignment.values()) + 1 if assignment else 0
        )

        canvas = tk.Canvas(
            self.table_panel,
            bg="white",
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            self.table_panel, orient="vertical",
            command=canvas.yview
        )

        inner = tk.Frame(canvas, bg="white")

        inner.bind(
            "<Configure>",
            lambda event: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window(
            (0, 0), window=inner, anchor="nw"
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        tk.Label(
            inner, text="Slot",
            bg="#E6EFFB", width=9,
            font=("Arial", 10, "bold"),
            relief="solid", borderwidth=1
        ).grid(row=0, column=0, sticky="nsew")

        for hall in range(self.r):
            tk.Label(
                inner,
                text=f"Hall {hall + 1}",
                bg="#E6EFFB", width=10,
                font=("Arial", 10, "bold"),
                relief="solid", borderwidth=1
            ).grid(row=0, column=hall + 1, sticky="nsew")

        for slot in range(slot_count):
            tk.Label(
                inner,
                text=f"Slot {slot + 1}",
                bg="#E6EFFB",
                font=("Arial", 10, "bold"),
                relief="solid", borderwidth=1
            ).grid(row=slot + 1, column=0, sticky="nsew")

            courses = [
                course for course, assigned_slot in assignment.items()
                if assigned_slot == slot
            ]
            courses.sort()

            for hall in range(self.r):
                course = courses[hall] if hall < len(courses) else "-"

                color = (
                    self.color_for_slot(slot)
                    if course != "-" else "#FFFFFF"
                )

                tk.Label(
                    inner,
                    text=str(course),
                    bg=color,
                    fg="white" if course != "-" else "#7A8490",
                    width=10,
                    height=2,
                    relief="solid",
                    borderwidth=1,
                    font=("Arial", 11, "bold")
                ).grid(
                    row=slot + 1,
                    column=hall + 1,
                    sticky="nsew"
                )

    def draw_animation_graph(self):
        for child in self.animation_panel.winfo_children():
            child.destroy()

        fig = Figure(figsize=(4.7, 2.7), dpi=90)
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        pos = nx.spring_layout(self.graph, seed=42, k=1.1)

        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            edge_color="#B6C0CC"
        )

        node_colors = []
        for course in self.graph.nodes():
            if course in self.assignment:
                node_colors.append(
                    self.color_for_slot(self.assignment[course])
                )
            else:
                node_colors.append("#DCE3EC")

        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors,
            node_size=420,
            edgecolors="#35475B"
        )

        nx.draw_networkx_labels(
            self.graph, pos, ax=ax, font_size=9
        )

        if self.animation_index > 0:
            last_course = self.animation_order[
                min(self.animation_index - 1,
                    len(self.animation_order) - 1)
            ]

            nx.draw_networkx_nodes(
                self.graph, pos, ax=ax,
                nodelist=[last_course],
                node_color=[self.color_for_slot(
                    self.assignment[last_course]
                )],
                node_size=620,
                edgecolors=NAVY,
                linewidths=3
            )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(
            fig, master=self.animation_panel
        )
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.animation_canvas = canvas

        if self.animation_index == 0:
            message = "Press Start Animation to begin."
        elif self.animation_index >= len(self.animation_order):
            message = "All courses have been assigned."
        else:
            message = (
                f"Assigned {self.animation_index}/"
                f"{len(self.animation_order)} courses"
            )

        tk.Label(
            self.animation_panel,
            text=message,
            bg="white",
            fg=NAVY,
            font=("Arial", 11, "bold")
        ).pack(pady=5)

    def draw_comparison(self):
        for child in self.chart_panel.winfo_children():
            child.destroy()

        fig = Figure(figsize=(5.7, 3.4), dpi=90)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        sizes = []
        greedy_slots = []
        exact_slots = []
        greedy_times = []
        exact_times = []

        # Use a fixed seed so the chart is reproducible.
        rng = random.Random(123)

        test_sizes = sorted(set([
            min(5, len(self.graph)),
            min(8, len(self.graph)),
            min(12, len(self.graph)),
            min(16, len(self.graph)),
            min(20, len(self.graph))
        ]))
        test_sizes = [size for size in test_sizes if size >= 2]

        for n in test_sizes:
            test_graph = nx.gnp_random_graph(
                n, 0.25, seed=rng.randint(1, 1_000_000)
            )
            test_graph = nx.relabel_nodes(
                test_graph,
                {node: node + 1 for node in test_graph.nodes()}
            )

            _, g_slots, g_time = greedy_coloring(
                test_graph, self.r
            )

            # Limit exact benchmarking to small graphs to avoid
            # freezing the GUI on hard instances.
            if n <= 12:
                g_assignment, _, _ = greedy_coloring(
                    test_graph, self.r
                )
                _, e_slots, e_time, _, _ = exact_coloring(
                    test_graph, self.r,
                    g_assignment, g_slots
                )
            else:
                e_slots = None
                e_time = None

            sizes.append(n)
            greedy_slots.append(g_slots)
            greedy_times.append(max(g_time, 0.000001))

            exact_slots.append(e_slots)
            exact_times.append(
                max(e_time, 0.000001)
                if e_time is not None else None
            )

        ax1.plot(
            sizes, greedy_slots, marker="o",
            color="#2677F5", label="Welsh-Powell"
        )

        exact_valid = [
            (n, value) for n, value in zip(sizes, exact_slots)
            if value is not None
        ]

        if exact_valid:
            ax1.plot(
                [item[0] for item in exact_valid],
                [item[1] for item in exact_valid],
                marker="s", color="#F07828",
                label="Exact backtracking"
            )

        ax1.set_title("Slots Used")
        ax1.set_xlabel("Courses (N)")
        ax1.set_ylabel("Number of slots")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=7)

        ax2.plot(
            sizes, greedy_times, marker="o",
            color="#2677F5", label="Welsh-Powell"
        )

        exact_time_valid = [
            (n, value)
            for n, value in zip(sizes, exact_times)
            if value is not None
        ]

        if exact_time_valid:
            ax2.plot(
                [item[0] for item in exact_time_valid],
                [item[1] for item in exact_time_valid],
                marker="s", color="#F07828",
                label="Exact backtracking"
            )

        ax2.set_title("Running Time")
        ax2.set_xlabel("Courses (N)")
        ax2.set_ylabel("Seconds")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig, master=self.chart_panel
        )
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.chart_canvas = canvas

        tk.Label(
            self.chart_panel,
            text="Exact timing is measured only for N <= 12.",
            bg="white",
            fg="#68788B",
            font=("Arial", 9)
        ).pack(pady=2)

    # ------------------ ANIMATION ------------------

    def start_animation(self):
        if self.animation_running:
            return

        if self.animation_index >= len(self.animation_order):
            self.reset_animation()

        self.animation_running = True
        self.animate_next()

    def animate_next(self):
        if not self.animation_running:
            return

        if self.animation_index >= len(self.animation_order):
            self.animation_running = False
            self.status.config(
                text="Animation complete. All courses assigned.",
                fg="#16884D"
            )
            return

        course = self.animation_order[self.animation_index]

        # Assign this course to the slot used by the greedy algorithm.
        # This makes the animation trace a real greedy assignment order.
        self.assignment[course] = self.greedy_assignment[course]
        slot = self.assignment[course] + 1

        self.animation_index += 1

        self.status.config(
            text=(
                f"Step {self.animation_index}/{len(self.animation_order)}: "
                f"Course {course} assigned to Slot {slot}"
            ),
            fg=NAVY
        )

        self.draw_all()
        self.after_id = self.root.after(850, self.animate_next)

    def pause_animation(self):
        self.animation_running = False

        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        self.status.config(text="Animation paused.", fg="#B35A00")

    def reset_animation(self):
        self.animation_running = False

        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        self.animation_index = 0
        self.assignment = {}
        self.status.config(text="Animation reset.", fg=NAVY)
        self.draw_all()

    def show_final(self):
        self.pause_animation()
        self.assignment = self.exact_assignment.copy()
        self.animation_index = len(self.animation_order)
        self.draw_all()

        self.status.config(
            text=(
                f"Greedy: {self.greedy_slots} slots | "
                f"Exact: {self.exact_slots} slots | "
                f"Exact optimality proven: {self.optimal}"
            ),
            fg="#16884D"
        )

    def export_csv(self):
        from tkinter.filedialog import asksaveasfilename
        import csv

        assignment = self.assignment or self.exact_assignment
        if not assignment:
            messagebox.showerror("Error", "No timetable available.")
            return

        slot_count = max(assignment.values()) + 1
        filename = asksaveasfilename(
            title="Save timetable CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="exam_timetable_visualization.csv"
        )

        if not filename:
            return

        try:
            with open(filename, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Slot",
                    *[f"Hall {hall}" for hall in range(1, self.r + 1)]
                ])

                for slot in range(slot_count):
                    courses = sorted(
                        course for course, assigned_slot in assignment.items()
                        if assigned_slot == slot
                    )

                    writer.writerow(
                        [slot + 1] + courses +
                        [""] * (self.r - len(courses))
                    )

            messagebox.showinfo(
                "Export successful",
                f"Timetable saved to:\n{filename}"
            )

        except OSError as error:
            messagebox.showerror("Export error", str(error))


# ------------------ START APPLICATION ------------------

def main():
    try:
        graph, r = read_graph()
    except (ValueError, nx.NetworkXError) as error:
        print("Input error:", error)
        return

    root = tk.Tk()

    try:
        TimetableDashboard(root, graph, r)
    except Exception as error:
        root.destroy()
        messagebox.showerror("Error", str(error))
        return

    root.mainloop()


if __name__ == "__main__":
    main()
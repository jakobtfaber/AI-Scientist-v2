"""
AI Scientist Live TUI Dashboard 📟
Powered by Textual & Rich

Usage:
    python dashboard_tui.py

Features:
- Real-time experiment status monitoring
- Live Perplexity research and reasoning display
- Validation metrics tracking
- Scrolling log viewer
"""

import time
import re
import os
from collections import deque
from datetime import datetime
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.console import Console
from rich import box
from rich.style import Style

# Configuration
LOG_FILE = "gpu_ffa_validation_test.log"
REFRESH_RATE = 4  # Hz

# State State
class DashboardState:
    def __init__(self):
        self.stage = "Initializing..."
        self.substage = "Wait..."
        self.goals = []
        self.metrics = {"CPU": "N/A", "GPU": "N/A", "Speedup": "N/A"}
        self.last_update = datetime.now()
        self.research_summary = "*Waiting for Perplexity research...*"
        self.validations = deque(maxlen=10)  # (Time, Check, Result, Message)
        self.logs = deque(maxlen=15)
        
        # Determine start time from log file creation if possible, otherwise now
        if os.path.exists(LOG_FILE):
             self.start_time = datetime.fromtimestamp(os.path.getctime(LOG_FILE))
        else:
             self.start_time = datetime.now()
             
        self.ai_reasoning = ""

    def add_log(self, line):
        clean_line = line.strip()
        if clean_line:
            self.logs.append(clean_line)
            self._parse_line(clean_line)

    def _parse_line(self, line):
        clean_line = line.strip()
        # Stage detection
        if "Current Main Stage:" in line:
            self.stage = line.split(":", 1)[1].strip()
        elif "Starting main stage:" in line:
            self.stage = line.split(":", 1)[1].strip()
        elif "Sub-stage:" in line:
            self.substage = line.split(":", 1)[1].strip()
        elif "Starting sub-stage:" in line:
            self.substage = line.split(":", 1)[1].strip()
            
        # Metric detection - regex based for robustness
        # Matches "CPU runtime: 123.4567s" or variations
        if "CPU runtime" in line:
            m = re.search(r"CPU runtime.*?(\d+\.\d+)", line)
            if m and "{" not in line:
                self.metrics["CPU"] = m.group(1) + "s"
                
        if "GPU runtime" in line:
            m = re.search(r"GPU runtime.*?(\d+\.\d+)", line)
            if m and "{" not in line:
                self.metrics["GPU"] = m.group(1) + "s"
                
        if "Speedup Factor" in line:
            m = re.search(r"Speedup Factor.*?(\d+\.\d+)", line)
            if m and "{" not in line:
                self.metrics["Speedup"] = m.group(1) + "x"

        # Validation detection
        if "DATA QUALITY" in line:
            # Example: [DATA QUALITY] [PASS] statistical_validity: 3 samples (CV=0.01)
            parts = line.split("]", 2)
            if len(parts) >= 3:
                result = parts[1].replace("[", "").strip()
                msg = parts[2].strip()
                self.validations.appendleft((datetime.now().strftime("%H:%M:%S"), "Data Quality", result, msg))
        
        # Reasoning detection
        if "Perplexity reasoning validation:" in line or "Reasoning:" in line:
             if "WARNING" in line and "Reasoning:" in line:
                 # Clean up the log line if it's the "WARNING Reasoning: ..." format
                 msg = line.split("Reasoning:", 1)[1].strip()
                 self.validations.appendleft((datetime.now().strftime("%H:%M:%S"), "AI Reasoning", "INFO", "Reasoning: " + msg[:50] + "..."))
             else:
                 self.validations.appendleft((datetime.now().strftime("%H:%M:%S"), "AI Reasoning", "INFO", "Reasoning triggered"))

state = DashboardState()

def make_header():
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="right", ratio=1)
    
    elapsed = datetime.now() - state.start_time
    elapsed_str = str(elapsed).split('.')[0]
    
    title = Text(" 🔬 The AI Scientist ", style="bold white on blue")
    status = Text(f"Status: RUNNING | Time: {elapsed_str}", style="bold green")
    
    grid.add_row(title, status, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return Panel(grid, style="white on black", box=box.HEAVY)

def make_status_panel():
    grid = Table.grid(expand=True)
    grid.add_column()
    
    # Status Table
    table = Table(box=None, expand=True, show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")
    
    table.add_row("Current Stage", state.stage)
    table.add_row("Sub-stage", state.substage)
    
    status_panel = Panel(table, title="[bold blue]Experiment Phase", border_style="blue")
    
    # Metrics Table
    m_table = Table(box=None, expand=True, show_header=True)
    m_table.add_column("Metric", style="yellow")
    m_table.add_column("Value", style="bold yellow")
    
    m_table.add_row("CPU Runtime", state.metrics["CPU"])
    m_table.add_row("GPU Runtime", state.metrics["GPU"])
    m_table.add_row("Speedup", state.metrics["Speedup"])
    
    metrics_panel = Panel(m_table, title="[bold yellow]Performance Metrics", border_style="yellow")
    
    # Goals Panel
    goals_text = "\n".join([f"• {g}" for g in state.goals]) if state.goals else "No active goals detected"
    goals_panel = Panel(goals_text, title="[bold white]Stage Goals", border_style="white")
    
    return Layout(
        Panel(
            Text(""), # Placeholder for structure/grid
            title="Overview",
            border_style="dim"
        )
    )

def make_left_column():
    # Manual Grid Layout for Left Column
    layout = Layout()
    layout.split_column(
        Layout(name="phase", size=6),
        Layout(name="metrics", size=6),
        Layout(name="goals", size=8),
        Layout(name="validation")
    )
    
    # Phase
    table = Table(box=None, expand=True, show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")
    table.add_row("Current Stage", state.stage)
    table.add_row("Sub-stage", state.substage)
    layout["phase"].update(Panel(table, title="[bold blue]Experiment Phase", border_style="blue", box=box.ROUNDED))
    
    # Metrics
    m_table = Table(box=None, expand=True, show_header=False)
    m_table.add_column("Metric", style="yellow")
    m_table.add_column("Value", style="bold yellow")
    m_table.add_row("CPU Runtime", state.metrics["CPU"])
    m_table.add_row("GPU Runtime", state.metrics["GPU"])
    m_table.add_row("Speedup", state.metrics["Speedup"])
    layout["metrics"].update(Panel(m_table, title="[bold yellow]Performance Metrics", border_style="yellow", box=box.ROUNDED))
    
    # Goals
    goals_text = "\n".join([f"• {g}" for g in state.goals]) if state.goals else "Reading goals..."
    layout["goals"].update(Panel(goals_text, title="[bold white]Current Goals", border_style="white", box=box.ROUNDED))
    
    # Validation
    v_table = Table(box=box.SIMPLE, expand=True)
    v_table.add_column("Time", style="dim")
    v_table.add_column("Type")
    v_table.add_column("Result")
    v_table.add_column("Message")
    
    if not state.validations:
        v_table.add_row("-", "-", "-", "No validation events yet")
    else:
        for t, type_, res, msg in state.validations:
            color = "green" if "PASS" in res else "red" if "FAIL" in res else "yellow"
            v_table.add_row(t, type_, f"[{color}]{res}[/{color}]", msg)
            
    layout["validation"].update(Panel(v_table, title="[bold magenta]Validation & Reasoning", border_style="magenta", box=box.ROUNDED))
    
    return layout

def make_research_panel():
    return Panel(
        Markdown(state.research_summary),
        title="[bold green]Perplexity Research Insights",
        border_style="green",
        box=box.ROUNDED
    )

def make_log_panel():
    log_text = Text()
    for line in state.logs:
        log_text.append(line + "\n")
        
    return Panel(
        log_text,
        title="[bold dim]Live Logs",
        border_style="white",
        box=box.ROUNDED
    )

def make_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body")
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1)
    )
    
    # Update components
    layout["header"].update(make_header())
    layout["left"].update(make_left_column())
    
    layout["right"].split_column(
        Layout(name="research", ratio=2),
        Layout(name="logs", ratio=1)
    )
    layout["right"]["research"].update(make_research_panel())
    layout["right"]["logs"].update(make_log_panel())
    
    return layout

# Variables for multi-line capture
research_buffer = []
capturing_research = False
metrics_buffer = []  # Unused, but kept for pattern
goals_buffer = []
capturing_goals = False

def update_loop():
    global capturing_research, capturing_goals
    
    # Determine offset
    with open(LOG_FILE, 'r') as f:
        f.seek(0, 2)
        last_pos = f.tell()
        # Backtrack further to catch goals/research if restarting
        if last_pos > 20000:
             f.seek(max(0, last_pos - 20000))
             last_pos = f.tell()
        else:
             f.seek(0)
             last_pos = 0

    with Live(make_layout(), refresh_per_second=REFRESH_RATE, screen=True) as live:
        while True:
            with open(LOG_FILE, 'r') as f:
                f.seek(last_pos)
                new_data = f.read()
                last_pos = f.tell()
            
            if new_data:
                for line in new_data.splitlines():
                    state.add_log(line)
                    stripped = line.strip()
                    
                    # --- Research Parsing ---
                    if "Research summary:" in line:
                        capturing_research = True
                        research_buffer.clear()
                    elif "Enhancing idea with Perplexity research..." in line:
                         state.research_summary = "*Researching...*"
                    
                    # Heuristic end of research block
                    if capturing_research:
                        if "========" in line or "Testing idea..." in line:
                            capturing_research = False
                            if research_buffer:
                                state.research_summary = "\n".join(research_buffer)
                        elif "Research summary:" not in line and len(stripped) > 0:
                            research_buffer.append(stripped)
                            # Live update
                            state.research_summary = "\n".join(research_buffer)

                    # --- Goal Parsing ---
                    if "Goals:" in line or "Sub-stage goals:" in line:
                        capturing_goals = True
                        goals_buffer.clear()
                    elif capturing_goals:
                         if stripped.startswith("-") or stripped.startswith("*"):
                             goal_text = stripped.lstrip("-* ").strip()
                             if goal_text:
                                 goals_buffer.append(goal_text)
                                 state.goals = list(goals_buffer)
                         elif len(stripped) > 0 and not stripped.startswith("-") and "Stage" not in line:
                             # End of indented list?
                             capturing_goals = False

            live.update(make_layout())
            time.sleep(1/REFRESH_RATE)

if __name__ == "__main__":
    try:
        update_loop()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

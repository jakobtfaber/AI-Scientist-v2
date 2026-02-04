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
        self.speedup_history = deque(maxlen=20) # Store float values
        self.last_update = datetime.now()
        self.research_summary = "*Waiting for Perplexity research...*"
        self.validations = deque(maxlen=10)  # (Time, Check, Result, Message)
        self.validation_stats = {"PASS": 0, "FAIL": 0}
        self.logs = deque(maxlen=30) # Increased log buffer
        
        # Identity
        self.title = "Unknown Experiment"
        self.hypothesis = "Waiting for definition..."
        
        # Determine start time from log file creation if possible, otherwise now
        if os.path.exists(LOG_FILE):
             self.start_time = datetime.fromtimestamp(os.path.getctime(LOG_FILE))
        else:
             self.start_time = datetime.now()
             
        self.ai_reasoning = ""
        self.experiment_dir = None

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
                val = float(m.group(1))
                self.metrics["Speedup"] = f"{val:.1f}x"
                self.speedup_history.append(val)

        # Validation detection
        if "DATA QUALITY" in line:
            # Example: [DATA QUALITY] [PASS] statistical_validity: 3 samples (CV=0.01)
            parts = line.split("]", 2)
            if len(parts) >= 3:
                result = parts[1].replace("[", "").strip()
                msg = parts[2].strip()
                self.validations.appendleft((datetime.now().strftime("%H:%M:%S"), "Data Quality", result, msg))
                if "PASS" in result:
                    self.validation_stats["PASS"] += 1
                elif "FAIL" in result:
                    self.validation_stats["FAIL"] += 1
        
        # Reasoning detection
        if "Perplexity reasoning validation:" in line or "Reasoning:" in line:
             if "WARNING" in line and "Reasoning:" in line:
                 # Clean up the log line if it's the "WARNING Reasoning: ..." format
                 msg = line.split("Reasoning:", 1)[1].strip()
                 self.validations.appendleft((datetime.now().strftime("%H:%M:%S"), "AI Reasoning", "INFO", "Reasoning: " + msg[:50] + "..."))
             else:
                 self.validations.appendleft((datetime.now().strftime("%H:%M:%S"), "AI Reasoning", "INFO", "Reasoning triggered"))

        # Experiment Dir detection
        if "Results will be saved in" in line:
            # Example: Results will be saved in experiments/2026-02-03_...
            parts = line.split("experiments/")
            if len(parts) > 1:
                self.experiment_dir = os.path.join("experiments", parts[1].strip())

state = DashboardState()

def make_sparkline(data, width=30):
    if not data:
        return Text("Waiting for data...", style="dim")
    
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    chars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    result = []
    
    for val in data:
        normalized = (val - min_val) / range_val
        idx = int(normalized * (len(chars) - 1))
        # Ensure index in bounds
        idx = max(0, min(idx, len(chars)-1))
        result.append(chars[idx])
        
    return Text("".join(result), style="bold yellow")

def make_header():
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right", ratio=1)
    
    elapsed = datetime.now() - state.start_time
    elapsed_str = str(elapsed).split('.')[0]
    
    # Determine Status Color
    if "complete" in state.stage.lower():
        color = "green"
        status_txt = "COMPLETE"
    else:
        color = "blue"
        status_txt = "RUNNING"
        
    title_text = Text(" 🔬 The AI Scientist", style=f"bold white on {color}")
    annot = Text(f" {state.title[:50]}... ", style="italic white on black")
    
    status_line = Text(f"{status_txt} | Runtime: {elapsed_str} ", style=f"bold {color}")
    
    grid.add_row(title_text + annot, status_line)
    return Panel(grid, style="white on black", box=box.HEAVY)

def make_left_column():
    # Manual Grid Layout for Left Column
    layout = Layout()
    layout.split_column(
        Layout(name="identity", size=7),  # Reduced
        Layout(name="metrics", size=8),   # Reduced
        Layout(name="goals", size=6),     # Reduced
        Layout(name="validation")         # Takes remaining space
    )
    
    # Identity Panel
    grid = Table.grid(expand=True)
    grid.add_row(Text("Project:", style="bold cyan"), Text(state.title, style="white"))
    grid.add_row(Text("Phase:", style="bold cyan"), Text(f"{state.stage}", style="yellow"))
    # Truncate hypothesis more aggressively
    hyp = state.hypothesis if len(state.hypothesis) < 60 else state.hypothesis[:57] + "..."
    grid.add_row(Text("Hypothesis:", style="bold cyan"), Text(hyp, style="dim white"))
    layout["identity"].update(Panel(grid, title="[bold blue]Project Context", border_style="blue", box=box.ROUNDED))
    
    # Metrics with Chart
    metrics_layout = Layout()
    metrics_layout.split_column(
        Layout(name="table", ratio=1),
        Layout(name="chart", size=3)
    )
    
    m_table = Table(box=None, expand=True)
    m_table.add_column("Metric", style="yellow")
    m_table.add_column("Value", style="bold yellow")
    m_table.add_row("CPU", state.metrics["CPU"])
    m_table.add_row("GPU", state.metrics["GPU"])
    m_table.add_row("Speedup", state.metrics["Speedup"])
    
    metrics_layout["table"].update(m_table)
    
    # Sparkline
    spark = make_sparkline(list(state.speedup_history))
    metrics_layout["chart"].update(Panel(spark, title="Speedup Trend", border_style="yellow"))

    layout["metrics"].update(Panel(metrics_layout, title="[bold yellow]Performance Analysis", border_style="yellow", box=box.ROUNDED))
    
    # Goals
    goals_text = "\n".join([f"• {g}" for g in state.goals]) if state.goals else "No active goals detected"
    layout["goals"].update(Panel(goals_text, title="[bold white]Current Goals", border_style="white", box=box.ROUNDED))
    
    # Validation
    v_stats_grid = Table.grid(expand=True)
    v_stats_grid.add_column(ratio=1)
    v_stats_grid.add_column(ratio=1)
    v_stats_grid.add_row(
        Text(f"PASS: {state.validation_stats['PASS']}", style="bold green"),
        Text(f"FAIL: {state.validation_stats['FAIL']}", style="bold red")
    )
    
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
            
    v_group = Table.grid(expand=True)
    v_group.add_row(v_stats_grid)
    v_group.add_row(v_table)
    
    layout["validation"].update(Panel(v_group, title="[bold magenta]Validation & Reasoning", border_style="magenta", box=box.ROUNDED))
    
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
title_buffer = False
hypothesis_buffer = False

def update_loop():
    global capturing_research, capturing_goals, title_buffer, hypothesis_buffer
    
    # Initial full scan to get current state
    last_pos = 0
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            # Read everything cleanly first
            content = f.read()
            last_pos = f.tell()
            for line in content.splitlines():
                state.add_log(line)
                # Run the state update logic on historical lines too
                _process_line_state(line.strip(), line)

    with Live(make_layout(), refresh_per_second=REFRESH_RATE, screen=True) as live:
        while True:
            # Update research from file if needed
            if state.experiment_dir and len(state.research_summary) < 50:
                 idea_file = os.path.join(state.experiment_dir, "idea.md")
                 if os.path.exists(idea_file):
                     try:
                         with open(idea_file, 'r') as f:
                             content = f.read()
                             if "## Perplexity Research" in content:
                                 # Extract just the research section
                                 research_text = content.split("## Perplexity Research")[1].strip()
                                 state.research_summary = research_text
                             elif len(content) > 50:
                                 state.research_summary = content
                     except Exception:
                         pass

            with open(LOG_FILE, 'r') as f:
                f.seek(last_pos)
                new_data = f.read()
                last_pos = f.tell()
            
            if new_data:
                for line in new_data.splitlines():
                    state.add_log(line)
                    _process_line_state(line.strip(), line)

            live.update(make_layout())
            time.sleep(1/REFRESH_RATE)

def _process_line_state(stripped, raw_line):
    global capturing_research, capturing_goals, research_buffer, goals_buffer
    global title_buffer, hypothesis_buffer
    
    # --- Identity Parsing ---
    if "Title:" in raw_line and len(stripped) < 10: # Just "Title:" on line
        title_buffer = True
    elif title_buffer:
        if stripped:
            state.title = stripped
            title_buffer = False
            
    if "Short Hypothesis:" in raw_line:
        hypothesis_buffer = True
    elif hypothesis_buffer:
        if stripped:
            state.hypothesis = stripped
            hypothesis_buffer = False

    # --- Research Parsing ---
    if "Research summary:" in stripped:
        capturing_research = True
        research_buffer.clear()
    elif "Enhancing idea with Perplexity research..." in stripped:
         state.research_summary = "*Researching...*"
    
    # Heuristic end of research block
    if capturing_research:
        if "========" in stripped or "Testing idea..." in stripped:
            capturing_research = False
            if research_buffer:
                state.research_summary = "\n".join(research_buffer)
        elif "Research summary:" not in stripped and len(stripped) > 0:
            research_buffer.append(stripped)
            # Live update
            state.research_summary = "\n".join(research_buffer)

    # --- Goal Parsing ---
    # Be robust to variations
    if "Goals:" in stripped or "Sub-stage goals:" in stripped:
        capturing_goals = True
        goals_buffer.clear()
    elif capturing_goals:
         if stripped.startswith("-") or stripped.startswith("*"):
             goal_text = stripped.lstrip("-* ").strip()
             if goal_text:
                 goals_buffer.append(goal_text)
                 state.goals = list(goals_buffer)
         elif len(stripped) > 0 and not stripped.startswith("-") and not stripped.startswith("*") and "Stage" not in stripped:
             # End of indented list?
             capturing_goals = False

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="AI Scientist TUI Dashboard")
    parser.add_argument("log_file", nargs="?", default="gpu_ffa_validation_test.log", help="Path to the log file to monitor")
    args = parser.parse_args()
    
    LOG_FILE = args.log_file
    
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file '{LOG_FILE}' not found.")
        print("Usage: python dashboard_tui.py [path/to/logfile.log]")
        sys.exit(1)

    try:
        update_loop()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        console.print_exception()

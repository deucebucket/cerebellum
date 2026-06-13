#!/bin/bash
# Waits for the current SSH-bound Claude session to exit, then resumes it
# inside tmux on the desktop so Remote Control can reach it from the phone.
OLD_PID=1215533
SESSION_ID=77d599d1-73bb-42dc-8cfc-f5ec49c7aa3d
LOG=/var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/session_handoff.log
echo "[$(date '+%F %T')] handoff armed: waiting for claude pid $OLD_PID to exit" >> "$LOG"
while kill -0 $OLD_PID 2>/dev/null; do sleep 20; done
echo "[$(date '+%F %T')] old session gone; waiting 15s for transcript flush" >> "$LOG"
sleep 15
tmux new-session -d -s cerebellum -c /var/home/deucebucket/ai-drive/cerebellum \
  "/home/deucebucket/.local/bin/claude --resume $SESSION_ID"
echo "[$(date '+%F %T')] resumed in tmux session 'cerebellum' (attach: tmux attach -t cerebellum; or via Remote Control on the phone)" >> "$LOG"

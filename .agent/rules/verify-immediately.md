---
trigger: always_on
---

When running a command in the CLI, never buffer the output, always set it to stream live into the terminal. After running a command, set short 10 second timeouts, after which you continuously analyze the output to ensure that processes do not hang or produce undesirable results. Do not detach from the session while a command is running, ever.

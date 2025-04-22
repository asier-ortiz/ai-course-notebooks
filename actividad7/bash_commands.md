
# Bash Commands Guide

## Basic Commands

| Command | Description |
|--------|-------------|
| `pwd` | Show the current working directory |
| `ls` | List files and directories |
| `cd <dir>` | Change directory |
| `mkdir <dir>` | Create a new directory |
| `touch <file>` | Create an empty file |
| `cp <source> <dest>` | Copy file or directory |
| `mv <source> <dest>` | Move or rename file or directory |
| `rm <file>` | Remove file |
| `rm -r <dir>` | Remove directory recursively |
| `cat <file>` | Display file content |
| `less <file>` | View file content page-by-page |
| `head <file>` | Show the first 10 lines of a file |
| `tail <file>` | Show the last 10 lines of a file |

## File Permissions

| Command | Description |
|--------|-------------|
| `chmod <mode> <file>` | Change permissions |
| `chown <user>:<group> <file>` | Change owner and group |
| `ls -l` | List files with permissions |

## Process Management

| Command | Description |
|--------|-------------|
| `ps aux` | Show running processes |
| `top` | Real-time system monitoring |
| `htop` | Interactive process viewer (needs installation) |
| `kill <pid>` | Kill process by PID |
| `killall <process_name>` | Kill all processes by name |

## Networking

| Command | Description |
|--------|-------------|
| `ping <host>` | Test connectivity to host |
| `curl <url>` | Transfer data from URL |
| `wget <url>` | Download files from URL |
| `ssh <user>@<host>` | Connect to remote host via SSH |
| `scp <source> <user>@<host>:<dest>` | Secure copy between hosts |

## Disk Usage

| Command | Description |
|--------|-------------|
| `df -h` | Show disk space usage |
| `du -sh <dir>` | Show size of a directory |

## Text Processing

| Command | Description |
|--------|-------------|
| `grep <pattern> <file>` | Search for pattern in file |
| `awk '{print $1}' <file>` | Print the first column of each line |
| `sed 's/foo/bar/' <file>` | Replace text in a file |
| `sort <file>` | Sort lines of a file |
| `uniq <file>` | Filter unique lines |
| `cut -d',' -f1 <file>` | Cut by delimiter (e.g., comma) |

## Scripting

| Command | Description |
|--------|-------------|
| `bash <script.sh>` | Run a bash script |
| `source <script.sh>` | Execute commands from script in the current shell |

## Package Management (Debian/Ubuntu)

| Command | Description |
|--------|-------------|
| `sudo apt update` | Update package lists |
| `sudo apt upgrade` | Upgrade installed packages |
| `sudo apt install <package>` | Install a package |
| `sudo apt remove <package>` | Remove a package |

## Advanced Tips

- **Command chaining**: `cmd1 && cmd2` (run cmd2 only if cmd1 succeeds)
- **Background jobs**: `cmd &`
- **View command history**: `history`
- **Search command history**: `Ctrl + r`
- **Redirect output**: `>` (overwrite) and `>>` (append)

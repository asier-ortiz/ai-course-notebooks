# Bash Scripting Guide

A concise and practical introduction to Bash scripting.

---

## Table of Contents
1. [What is Bash?](#what-is-bash)
2. [Hello World](#hello-world)
3. [Variables](#variables)
4. [Conditionals](#conditionals)
5. [Loops](#loops)
6. [Functions](#functions)
7. [Common Flags](#common-flags)
8. [Working with Files](#working-with-files)
9. [Arrays](#arrays)
10. [Error Handling & Debugging](#error-handling--debugging)
11. [Traps and Signals](#traps-and-signals)
12. [Environment Configuration](#environment-configuration)
13. [Helper Functions](#helper-functions)
14. [Waiting for Readiness](#waiting-for-readiness)
15. [Logging Output](#logging-output)
16. [Tips](#tips)

---

## What is Bash?

**Bash** (Bourne Again SHell) is a Unix shell and command language used for scripting and command-line interaction. It is widely used for automating tasks on Unix-like systems.

---

## Hello World

```bash
#!/bin/bash

echo "Hello, world!"
```

Make it executable:
```bash
chmod +x script.sh
./script.sh
```

---

## Variables

```bash
name="Alice"
echo "Hello, $name"
```

- Use `"$variable"` to preserve spaces.
- No spaces around the `=` sign.
- Default value syntax: `${VAR:-default}` → use `default` if `VAR` is unset or empty.

---

## Conditionals

```bash
if [ "$name" == "Alice" ]; then
    echo "Welcome, Alice!"
elif [ -z "$name" ]; then
    echo "Name is empty."
else
    echo "Who are you?"
fi
```

Common tests:
- `-z "$var"` → True if variable is empty.
- `-n "$var"` → True if variable is not empty.
- `-f file.txt` → File exists and is a regular file.
- `-d dir/` → Directory exists.

---

## Loops

### For Loop
```bash
for i in 1 2 3; do
    echo "Number: $i"
done
```

### While Loop
```bash
count=1
while [ $count -le 5 ]; do
    echo "Count: $count"
    ((count++))
done
```

---

## Functions

```bash
say_hello() {
    echo "Hello, $1!"
}

say_hello "Bob"
```

- `$1`, `$2`, etc. are positional arguments.

---

## Common Flags

- `-z` → String is empty
- `-n` → String is not empty
- `-f` → File exists
- `-d` → Directory exists
- `==` or `=` → String comparison

Use `[[ ... ]]` for more advanced expressions.

---

## Working with Files

### Reading a file line-by-line
```bash
while IFS= read -r line; do
    echo "$line"
done < file.txt
```

### Writing to a file
```bash
echo "Hello" > file.txt   # Overwrites
echo "World" >> file.txt  # Appends
```

---

## Arrays

```bash
fruits=("apple" "banana" "cherry")
echo "First fruit: ${fruits[0]}"
echo "All fruits: ${fruits[@]}"
echo "Number of fruits: ${#fruits[@]}"
```

Looping through an array:
```bash
for fruit in "${fruits[@]}"; do
    echo "$fruit"
done
```

---

## Error Handling & Debugging

### Exit on Error
```bash
set -e  # Exit on first error
```

### Debug Mode
```bash
bash -x script.sh  # Prints each command before executing
```

### Check Command Status
```bash
if ! command; then
    echo "Command failed"
fi
```

---

## Traps and Signals

Trap cleanup or custom actions on exit or interruption:
```bash
trap 'echo "Cleaning up..."; rm -f temp.txt' EXIT
```

Trap Ctrl+C (SIGINT):
```bash
trap 'echo "Script interrupted."; exit 1' INT
```

---

## Environment Configuration

### Loading Variables from a .env File
```bash
if [ -f ".env" ]; then
  while IFS='=' read -r key value; do
    if [[ -n "$key" && "$key" != \#* ]]; then
      export "$key=$value"
    fi
  done < .env
fi
```

---

## Helper Functions

Encapsulate repetitive logic for clarity and reuse:
```bash
file_exists() {
  [ -f "$1" ]
}

dir_exists() {
  [ -d "$1" ]
}
```

Usage:
```bash
if file_exists "output.txt"; then
  echo "File found."
fi
```

---

## Waiting for Readiness

Useful for waiting on services or files to be available:
```bash
until some_command; do
  echo "Waiting for resource..."
  sleep 5
done
```

---

## Logging Output

Redirect output to a log file:
```bash
./script.sh > script.log 2>&1
```

Append output instead of overwriting:
```bash
./script.sh >> script.log 2>&1
```

Use `tee` to log and print:
```bash
./script.sh | tee script.log
```

---

## Tips

- Always quote variables: `"$var"`
- Use `set -e` to stop on errors
- Use `#!/bin/bash` at the top of scripts
- Test scripts with `bash -x script.sh`
- Prefer `"$(command)"` over backticks `` `command` ``
- Use functions for readability and reuse
- Comment your code generously with `#`

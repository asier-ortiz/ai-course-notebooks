# HDFS Commands Cheatsheet

This is a collection of basic HDFS commands along with a brief explanation of each one.

---

## Table of Contents
1. [Directory Management](#directory-management)
2. [File Management](#file-management)
3. [Additional Information](#additional-information)
4. [Tips](#tips)

---

## Directory Management

- **Create a directory:**
  ```bash
  hdfs dfs -mkdir /path/to/directory
  ```
  > Creates a new directory in HDFS.

- **Create nested directories:**
  ```bash
  hdfs dfs -mkdir -p /parent/child
  ```
  > Creates multiple directories at once (including parent directories if they don't exist).

- **List contents:**
  ```bash
  hdfs dfs -ls /path
  ```
  > Lists the files and folders in the specified directory.

- **View detailed info:**
  ```bash
  hdfs dfs -ls -h /path
  ```
  > Displays file sizes in a human-readable format (e.g., MB, GB).

- **Remove an empty directory:**
  ```bash
  hdfs dfs -rmdir /path
  ```
  > Deletes a directory only if it is empty.

- **Remove a directory with contents:**
  ```bash
  hdfs dfs -rm -r /path
  ```
  > Deletes the directory and all files inside it.

---

## File Management

- **Upload a local file to HDFS:**
  ```bash
  hdfs dfs -put local_file /path/in/hdfs
  ```
  > Copies a local file to a directory in HDFS.

- **Download a file from HDFS to local:**
  ```bash
  hdfs dfs -get /path/in/hdfs/file /local/path
  ```
  > Copies a file from HDFS to your local filesystem.

- **Move a file within HDFS:**
  ```bash
  hdfs dfs -mv /source/path/file /destination/path
  ```
  > Moves (or renames) a file or folder within HDFS.

- **Delete a file:**
  ```bash
  hdfs dfs -rm /path/file
  ```
  > Deletes a file in HDFS.

- **View the contents of a file:**
  ```bash
  hdfs dfs -cat /path/file
  ```
  > Displays the contents of a file in the terminal.

- **Count files, directories, and bytes:**
  ```bash
  hdfs dfs -count /path
  ```
  > Shows the number of files, directories, and total data size.

---

## Additional Information

- **Check HDFS usage:**
  ```bash
  hdfs dfsadmin -report
  ```
  > Displays HDFS system statistics (nodes, used/free space, etc.).

- **Check space used by a directory:**
  ```bash
  hdfs dfs -du -h /path
  ```
  > Displays the size occupied by each file/directory within the specified path.

---

## Tips

- You can use `hdfs dfs -help` to see more options for any command.
